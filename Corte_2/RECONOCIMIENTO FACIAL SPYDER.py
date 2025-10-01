"""
drowsiness_project.py

Proyecto completo: detección de somnolencia (sleepy vs awake)
Diseñado para ejecutarse en Spyder en un entorno con:
- Python 3.10/3.11 (64-bit)
- numpy, opencv-python, pillow, mtcnn, mediapipe (opcional), tensorflow

Funciones principales:
- capture_and_label(): captura imágenes desde cámara y las etiqueta (s = sleepy, a = awake)
- prepare_generators(): organiza ImageDataGenerators para entrenamiento
- build_model(): MobileNetV2 transfer learning
- train_model(): entrena con callbacks y guarda checkpoints
- load_model(): carga modelo .h5
- infer_image(), infer_video(), live_predict(): inferencia sobre archivos y en vivo
- mediapipe_eye_ratio(): heurístico (si no hay modelo entrenado)

Uso general:
1) Recolecta datos: capture_and_label()
2) Revisa /content/dataset (por defecto './dataset/train/...') y ajusta si necesitas
3) Ejecuta train_model(...) para entrenar
4) Usa live_predict() para inferencia en webcam o infer_video() / infer_image()
"""

import os
import time
from pathlib import Path
from datetime import datetime
import zipfile
import shutil
import numpy as np
from PIL import Image
import cv2

# Import ML libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Optional libs
try:
    from mtcnn import MTCNN
    mtcnn_detector = MTCNN()
except Exception:
    mtcnn_detector = None

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    mp_face_mesh = None

# ---------------- CONFIG / DEPENDENCIAS ----------------
# Ajusta si quieres otras rutas / tamaños
BASE_DIR = Path.cwd()  # carpeta del proyecto
DATASET_DIR = BASE_DIR / "dataset"   # estructura: dataset/train/<class>/, dataset/val/<class>/ (opcional)
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
MODEL_SAVE_PATH = BASE_DIR / "model_suenio.h5"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------- utilidades de datos ----------------
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def ensure_dirs():
    (DATASET_DIR / 'train' / 'sleepy').mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / 'train' / 'awake').mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / 'val' / 'sleepy').mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / 'val' / 'awake').mkdir(parents=True, exist_ok=True)

def list_dataset_counts(root=DATASET_DIR):
    root = Path(root)
    info = {}
    if not root.exists():
        return info
    for split in ['train', 'val']:
        sp = root / split
        if not sp.exists():
            continue
        for cls in sp.iterdir():
            if cls.is_dir():
                n = sum(1 for f in cls.rglob('*') if f.suffix.lower() in IMAGE_EXTS)
                info[f"{split}/{cls.name}"] = n
    return info

def show_samples(root=DATASET_DIR, n_samples=3):
    import matplotlib.pyplot as plt
    root = Path(root)
    for split in ['train','val']:
        sp = root / split
        if not sp.exists():
            print(f"No existe {sp}")
            continue
        classes = [c for c in sp.iterdir() if c.is_dir()]
        if not classes:
            print(f"No hay clases en {sp}")
            continue
        rows = len(classes); cols = n_samples
        plt.figure(figsize=(cols*3, rows*3))
        i=1
        for c in classes:
            imgs = [p for p in c.rglob('*') if p.suffix.lower() in IMAGE_EXTS]
            samples = imgs[:n_samples] if len(imgs)>=n_samples else imgs + imgs[:max(0,n_samples-len(imgs))]
            for s in samples:
                try:
                    im = Image.open(s).convert('RGB')
                    plt.subplot(rows,cols,i); plt.imshow(im); plt.title(f"{c.name}\n{s.name}", fontsize=8); plt.axis('off')
                except Exception as e:
                    plt.subplot(rows,cols,i); plt.text(0.5,0.5,str(e)); plt.axis('off')
                i+=1
        plt.tight_layout(); plt.show()

# ---------------- detección y recorte de cara ----------------
def detect_and_crop_face(frame):
    """Intenta MTCNN primero, luego Haarcascade OpenCV como fallback. Devuelve BGR recortado o None."""
    # MTCNN detector
    if mtcnn_detector is not None:
        try:
            res = mtcnn_detector.detect_faces(frame)
            if res:
                res = sorted(res, key=lambda r: r['box'][2]*r['box'][3], reverse=True)[0]
                x,y,w,h = res['box']
                x,y = max(0,x), max(0,y)
                return frame[y:y+h, x:x+w]
        except Exception:
            pass

    # OpenCV Haarcascade fallback
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x,y,w,h = faces[0]
        x,y = max(0,x), max(0,y)
        return frame[y:y+h, x:x+w]
    except Exception:
        return None

def preprocess_face_for_model(face, size=IMG_SIZE):
    face = cv2.resize(face, size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    return np.expand_dims(face, axis=0)

# ---------------- heurístico MediaPipe ----------------
def mediapipe_eye_ratio(frame):
    """Calcula un ratio vertical/horizontal aproximado del ojo; usable como heurístico (mayor = más abierto)."""
    if mp is None or mp_face_mesh is None:
        return None
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        h,w = frame.shape[:2]
        def xy(idx): return np.array([lm.landmark[idx].x*w, lm.landmark[idx].y*h])
        top = xy(159); bottom = xy(145); left = xy(33); right = xy(133)
        vert = np.linalg.norm(top-bottom); hor = np.linalg.norm(left-right)
        if hor == 0: return None
        return float(vert/hor)

# ---------------- captura/etiquetado ----------------
def capture_and_label(camera_idx=0, save_root=DATASET_DIR / 'train', img_size=IMG_SIZE):
    """
    Abre la cámara:
      s -> guardar como sleepy
      a -> guardar como awake
      q -> salir
    """
    save_root = Path(save_root)
    (save_root / 'sleepy').mkdir(parents=True, exist_ok=True)
    (save_root / 'awake').mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Prueba otro índice o cierra apps que usan la cámara.")
        return

    print("Captura iniciada. s=sleepy, a=awake, q=salir")
    counters = {'sleepy':0, 'awake':0}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face = detect_and_crop_face(frame)
        display = frame.copy()
        if face is not None:
            small = cv2.resize(face, (160,160))
            h,w = display.shape[:2]
            sw, sh = small.shape[1], small.shape[0]
            display[5:5+sh, w-5-sw:w-5] = small

        cv2.putText(display, "s=sleepy  a=awake  q=quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Capture & Label", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') or key == ord('a'):
            label = 'sleepy' if key==ord('s') else 'awake'
            to_save = face if face is not None else frame
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"{label}_{ts}.jpg"
            p = save_root / label / fname
            # resize and save
            try:
                pil = Image.fromarray(cv2.cvtColor(to_save, cv2.COLOR_BGR2RGB))
                pil = pil.resize(img_size, Image.LANCZOS)
                pil.save(p)
                counters[label] += 1
                print("Guardada:", p)
            except Exception as e:
                print("Error guardando:", e)
    cap.release()
    cv2.destroyAllWindows()
    print("Captura terminada. Totales:", counters)

# ---------------- preparar generadores ----------------
def prepare_generators(dataset_root=DATASET_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    dataset_root = Path(dataset_root)
    train_dir = dataset_root / 'train'
    val_dir = dataset_root / 'val'
    if not train_dir.exists():
        raise FileNotFoundError(f"No existe {train_dir}. Crea datos con capture_and_label() u organiza dataset.")

    # Si no hay val, usamos validation_split
    if val_dir.exists() and any(val_dir.iterdir()):
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           rotation_range=20,
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           zoom_range=0.1,
                                           horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1./255)
        train_gen = train_datagen.flow_from_directory(str(train_dir), target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=True)
        val_gen = val_datagen.flow_from_directory(str(val_dir), target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)
    else:
        datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=20,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.1,
                                     horizontal_flip=True,
                                     validation_split=0.2)
        train_gen = datagen.flow_from_directory(str(train_dir), target_size=img_size, batch_size=batch_size, class_mode='binary', subset='training', shuffle=True)
        val_gen = datagen.flow_from_directory(str(train_dir), target_size=img_size, batch_size=batch_size, class_mode='binary', subset='validation', shuffle=False)
    return train_gen, val_gen

# ---------------- modelo ----------------
def build_model(input_shape=(160,160,3), lr=1e-4, dropout=0.3):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------------- entrenamiento ----------------
def train_model(model, train_gen, val_gen, epochs=20, save_path=MODEL_SAVE_PATH, checkpoint_dir=CHECKPOINT_DIR):
    checkpoint_cb = ModelCheckpoint(str(checkpoint_dir / 'ckpt_epoch_{epoch:03d}.h5'),
                                    monitor='val_loss', save_best_only=False, save_weights_only=False, verbose=1)
    best_cb = ModelCheckpoint(str(save_path), monitor='val_loss', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    steps = max(1, train_gen.samples // train_gen.batch_size)
    val_steps = max(1, val_gen.samples // val_gen.batch_size)

    history = model.fit(train_gen, epochs=epochs, steps_per_epoch=steps, validation_data=val_gen, validation_steps=val_steps, callbacks=[checkpoint_cb, best_cb, early, reduce_lr])
    print("Entrenamiento finalizado.")
    return history

# ---------------- inferencia ----------------
def infer_image(model, img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    face = detect_and_crop_face(img)
    if face is None:
        return {'error': 'No face detected'}
    x = preprocess_face_for_model(face)
    pred = float(model.predict(x)[0][0])
    return {'label': 'sleepy' if pred>=0.5 else 'awake', 'prob': pred}

def infer_video(model, video_path, out_path=BASE_DIR / 'annotated_output.mp4', frame_step=5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    preds = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            face = detect_and_crop_face(frame)
            if face is not None:
                x = preprocess_face_for_model(face)
                pred = float(model.predict(x)[0][0])
                preds.append(pred)
                label = 'SUEÑO' if pred>=0.5 else 'NO SUEÑO'
                cv2.putText(frame, f"{label} ({pred:.2f})", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                # draw rect
                # we will re-detect bounding box with mtcnn for annotation:
                if mtcnn_detector is not None:
                    try:
                        faces = mtcnn_detector.detect_faces(frame)
                        if faces:
                            x0,y0,w0,h0 = faces[0]['box']
                            cv2.rectangle(frame, (x0,y0),(x0+w0,y0+h0),(0,255,0),2)
                    except Exception:
                        pass
        writer.write(frame)
        frame_idx += 1
    cap.release(); writer.release()
    mean_pred = float(np.mean(preds)) if preds else None
    final_label = 'sleepy' if mean_pred is not None and mean_pred>=0.5 else 'awake'
    return {'label': final_label, 'probability': mean_pred, 'output_video': str(out_path)}

def live_predict(model=None, camera_idx=0, use_heuristic_if_no_model=True, closed_threshold=0.19):
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return
    print("Presiona q para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face = detect_and_crop_face(frame)
        label = 'No face'
        prob_text = ''
        if model is not None and face is not None:
            x = preprocess_face_for_model(face)
            try:
                pred = float(model.predict(x)[0][0])
                label = 'sleepy' if pred>=0.5 else 'awake'
                prob_text = f"{pred:.3f}"
            except Exception as e:
                label = 'model error'
                prob_text = str(e)
        elif face is not None and use_heuristic_if_no_model:
            ratio = mediapipe_eye_ratio(face) if mp_face_mesh is not None else None
            if ratio is None:
                label = 'unknown'
            else:
                label = 'sleepy' if ratio < closed_threshold else 'awake'
                prob_text = f"ratio={ratio:.3f}"

        disp = frame.copy()
        if face is not None:
            sw = int(disp.shape[1]*0.25)
            sh = int(disp.shape[0]*0.25)
            small = cv2.resize(face, (sw, sh))
            disp[5:5+small.shape[0], disp.shape[1]-5-small.shape[1]:disp.shape[1]-5] = small

        cv2.putText(disp, f"{label} {prob_text}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Live predict", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()

# ----------------- util: cargar el mejor modelo si existe -----------------
def load_best_model(path=MODEL_SAVE_PATH):
    path = Path(path)
    if not path.exists():
        print("No existe modelo en", path)
        return None
    try:
        model = load_model(str(path))
        print("Modelo cargado:", path)
        return model
    except Exception as e:
        print("Error cargando el modelo:", e)
        return None

# ---------------- menú CLI (para usar en Spyder) ----------------
def main_menu():
    ensure_dirs()
    print("=== Drowsiness Project - Menú ===")
    print("1) Capturar y etiquetar imágenes desde cámara")
    print("2) Mostrar resumen del dataset")
    print("3) Entrenar modelo (transfer learning MobileNetV2)")
    print("4) Cargar mejor modelo y hacer inferencia en vivo (webcam)")
    print("5) Inferir un video con modelo y guardar video anotado")
    print("6) Inferir imagen con modelo")
    print("7) Salir")
    choice = input("Elige opción (1-7): ").strip()
    if choice == '1':
        capture_and_label()
    elif choice == '2':
        info = list_dataset_counts()
        print("Conteo por clase:")
        for k,v in info.items():
            print(f" - {k}: {v}")
        show_samples()
    elif choice == '3':
        # preparar datos y entrenar
        train_gen, val_gen = prepare_generators()
        model = build_model(input_shape=(IMG_SIZE[0],IMG_SIZE[1],3))
        epochs = input("Número de epochs (enter=20): ").strip()
        epochs = int(epochs) if epochs else 20
        train_model(model, train_gen, val_gen, epochs=epochs)
    elif choice == '4':
        mp = input("Ruta a modelo .h5 (enter para usar default %s): " % MODEL_SAVE_PATH).strip()
        model = load_best_model(mp if mp else MODEL_SAVE_PATH)
        live_predict(model=model, camera_idx=0)
    elif choice == '5':
        mp = input("Ruta a modelo .h5 (enter para usar default %s): " % MODEL_SAVE_PATH).strip()
        model = load_best_model(mp if mp else MODEL_SAVE_PATH)
        if model is None:
            print("No hay modelo cargado.")
        else:
            vp = input("Ruta al video a inferir: ").strip()
            out = input("Ruta salida video anotado (enter para default annotated_output.mp4): ").strip()
            res = infer_video(model, vp, out_path=BASE_DIR / (out if out else 'annotated_output.mp4'))
            print(res)
    elif choice == '6':
        mp = input("Ruta a modelo .h5 (enter para usar default %s): " % MODEL_SAVE_PATH).strip()
        model = load_best_model(mp if mp else MODEL_SAVE_PATH)
        if model is None:
            print("No hay modelo cargado.")
        else:
            ip = input("Ruta a imagen: ").strip()
            r = infer_image(model, ip)
            print(r)
    else:
        print("Saliendo.")

if __name__ == "__main__":
    main_menu()
