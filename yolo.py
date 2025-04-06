# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from ultralytics import YOLO
import collections
import threading
import time
import os
import torch
import traceback # برای چاپ خطاهای دقیق‌تر

# --- تنظیمات کلی ---
# لیست مدل‌های استاندارد قابل انتخاب
AVAILABLE_MODELS = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
# مدل پیش‌فرض برای بارگذاری اولیه
DEFAULT_MODEL_NAME = 'yolov8n'
CONFIDENCE_THRESHOLD = 0.45
TARGET_CLASSES = None
INPUT_SIZE = 640
SKIP_FRAMES_FACTOR = 1

# --- متغیرهای سراسری ---
processing_active = False
stop_event = threading.Event()
yolo_thread = None
model = None
class_names = {}
DEVICE = 'cpu'
current_model_name = "" # نام مدل بارگذاری شده فعلی (مثلا yolov8n)
root = None
status_label = None
model_display_entry = None # Entry برای نمایش مدل فعلی
model_select_combo = None # Combobox برای انتخاب مدل
btn_load_model = None     # دکمه بارگذاری مدل انتخاب شده
btn_image = None
btn_video = None
btn_webcam = None
btn_stop = None
# btn_select_model حذف می‌شود

# =====================================================
#   تعریف تمام توابع قبل از استفاده
# =====================================================

def check_cuda():
    """بررسی وضعیت CUDA و تنظیم DEVICE."""
    global DEVICE
    # ... (کد check_cuda بدون تغییر) ...
    detected_device = 'cpu'
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"CUDA شناسایی شد. GPU: {gpu_name}")
                try:
                    tensor_on_gpu = torch.tensor([1.0, 2.0]).to('cuda')
                    print(f"  تست CUDA موفق.")
                    detected_device = 'cuda'
                except Exception as e_cuda_test:
                    print(f"  خطا در تست CUDA: {e_cuda_test}")
                    print("  --> به CPU بازگشت می‌شود.")
            else:
                print("CUDA در دسترس است اما device_count صفر است.")
        else:
            print("CUDA در دسترس نیست.")
    except Exception as e:
        print(f"خطای کلی در بررسی CUDA: {e}")
    finally:
        DEVICE = detected_device
        print(f"دستگاه پردازش نهایی: '{DEVICE}'")
        return DEVICE == 'cuda'


def load_yolo_model(model_filename_to_load):
    """
    مدل YOLO را از نام فایل (مثلا 'yolov8n.pt') بارگذاری می‌کند.
    اگر فایل وجود نداشته باشد، Ultralytics تلاش می‌کند آن را دانلود کند.
    Args:
        model_filename_to_load (str): نام فایل مدل (e.g., 'yolov8n.pt').
    Returns:
        bool: True اگر بارگذاری موفق بود، False در غیر این صورت.
    """
    global model, class_names, DEVICE, current_model_name

    if not root:
        print("خطای داخلی: ریشه Tkinter در load_yolo_model وجود ندارد.")
        return False

    if not model_filename_to_load or not isinstance(model_filename_to_load, str) or not model_filename_to_load.endswith('.pt'):
         print(f"خطا: نام فایل مدل نامعتبر است: {model_filename_to_load}")
         messagebox.showerror("خطای مدل", f"نام فایل مدل ارائه شده نامعتبر است:\n{model_filename_to_load}\nباید به '.pt' ختم شود.", parent=root)
         # وضعیت قبلی مدل را حفظ کن
         return False

    # فایل .pt ممکن است هنوز وجود نداشته باشد (برای دانلود)
    # if not os.path.exists(model_filename_to_load):
    #      print(f"هشدار: فایل مدل '{model_filename_to_load}' به صورت محلی یافت نشد. تلاش برای دانلود...")
         # نیازی به بررسی وجود فایل نیست، YOLO خودش مدیریت می‌کند

    # آزادسازی حافظه مدل قبلی
    if model is not None:
        print(f"آزادسازی مدل قبلی ({current_model_name}) از حافظه '{DEVICE}'...")
        try:
            del model
            if DEVICE == 'cuda':
                 torch.cuda.empty_cache()
            model = None
            class_names = {}
            current_model_name = "" # پاک کردن نام مدل قبلی
            print("مدل قبلی آزاد شد.")
        except Exception as e_del:
            print(f"خطا هنگام آزادسازی مدل قبلی: {e_del}")

    try:
        model_base_name = model_filename_to_load.replace('.pt', '') # مثل 'yolov8n'
        print(f"در حال بارگذاری مدل '{model_filename_to_load}'...")
        if status_label:
             # ممکن است شامل دانلود باشد
             update_status(f"در حال بارگذاری/دانلود مدل '{model_filename_to_load}'...")
             root.update_idletasks() # نمایش فوری پیام
        else:
             print("(Status label not available yet for update)")

        # *** این خط مدل را بارگذاری یا دانلود می‌کند ***
        model_instance = YOLO(model_filename_to_load)
        model_instance.to(DEVICE)
        print(f"مدل به دستگاه '{DEVICE}' منتقل شد.")

        print("انجام پیش‌بینی آزمایشی برای 'گرم کردن' مدل...")
        dummy_input = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
        _ = model_instance.predict(dummy_input, verbose=False, device=DEVICE)
        print(f"مدل '{model_filename_to_load}' با موفقیت روی دستگاه '{DEVICE}' بارگذاری و تست شد.")

        model = model_instance
        class_names = model.names
        current_model_name = model_base_name # ذخیره نام مدل فعلی (بدون .pt)
        if status_label:
             update_status(f"مدل '{current_model_name}' بارگذاری شد. دستگاه: {DEVICE.upper()}. آماده.")
        if model_display_entry:
             update_model_display() # نمایش نام مدل جدید در Entry
        return True

    except ImportError as e:
        print(f"خطای وارد کردن ماژول: {e}")
        traceback.print_exc()
        messagebox.showerror("خطای وابستگی", f"خطا در وارد کردن ماژول‌های مورد نیاز:\n{e}\n\nمطمئن شوید ultralytics نصب شده.", parent=root)
        model = None
        class_names = {}
        current_model_name = ""
        if model_display_entry: update_model_display("خطا در بارگذاری")
        if status_label: update_status("خطا در بارگذاری مدل (وابستگی).")
        return False
    except Exception as e: # می‌تواند شامل خطای دانلود، خطای بارگذاری و ... باشد
        print(f"خطا در بارگذاری/دانلود/تست مدل YOLO ('{model_filename_to_load}') روی دستگاه '{DEVICE}': {e}")
        traceback.print_exc()
        messagebox.showerror("خطای مدل", f"خطا در بارگذاری/دانلود مدل ('{model_filename_to_load}') روی دستگاه '{DEVICE}'.\n\n{e}\n\n اینترنت را بررسی کنید و کنسول را ببینید.", parent=root)
        model = None
        class_names = {}
        current_model_name = ""
        if model_display_entry: update_model_display("خطا در بارگذاری")
        if status_label: update_status(f"خطا در بارگذاری مدل '{model_filename_to_load}'.")
        return False
    finally:
        if root:
             root.after(10, enable_controls) # فعال/غیرفعال کردن دکمه‌ها پس از تلاش برای بارگذاری


def load_selected_model_from_combobox():
    """مدل انتخاب شده در Combobox را بارگذاری می‌کند."""
    if not root: return
    if processing_active:
        messagebox.showwarning("هشدار", "لطفاً قبل از تغییر مدل، پردازش فعلی را متوقف کنید.", parent=root)
        return

    selected_model_name = model_select_combo.get() # مانند 'yolov8n'
    if not selected_model_name:
        messagebox.showwarning("انتخاب مدل", "لطفاً یک مدل از لیست انتخاب کنید.", parent=root)
        return

    model_filename = selected_model_name + ".pt" # ساخت نام فایل: 'yolov8n.pt'
    print(f"کاربر درخواست بارگذاری مدل '{model_filename}' را داد.")

    # غیرفعال کردن کنترل‌ها قبل از شروع بارگذاری
    disable_controls_for_loading()

    # فراخوانی تابع اصلی بارگذاری
    load_yolo_model(model_filename)
    # وضعیت دکمه‌ها توسط finally در load_yolo_model تنظیم می‌شود.


def update_model_display(text_override=None):
    """متن ویجت Entry نمایش مدل بارگذاری شده را به‌روزرسانی می‌کند."""
    global model_display_entry, current_model_name
    if model_display_entry and model_display_entry.winfo_exists():
        try:
            model_display_entry.config(state=tk.NORMAL)
            model_display_entry.delete(0, tk.END)
            if text_override:
                model_display_entry.insert(0, text_override)
            elif current_model_name:
                 # فقط نام مدل (مثل yolov8n) را نمایش بده
                 model_display_entry.insert(0, f"{current_model_name} ({DEVICE.upper()})")
            else:
                 model_display_entry.insert(0, "هیچ مدلی بارگذاری نشده")
            model_display_entry.config(state='readonly')
        except tk.TclError as e:
            print(f"خطای Tkinter هنگام به‌روزرسانی نمایش مدل: {e}")


# --- توابع پردازش و نمایش ---
def process_predictions(results):
    """پردازش نتایج تشخیص YOLO."""
    # ... (کد process_predictions بدون تغییر) ...
    total_count = 0
    specific_counts = collections.defaultdict(int)
    if not class_names:
        return 0, specific_counts
    if results.boxes is not None:
        for box in results.boxes:
            if box.cls is not None and box.conf is not None and len(box.cls) > 0 and len(box.conf) > 0:
                try:
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    if confidence >= CONFIDENCE_THRESHOLD:
                        class_name = class_names.get(cls_id, f"ID:{cls_id}")
                        if TARGET_CLASSES is None or class_name in TARGET_CLASSES:
                            specific_counts[class_name] += 1
                            total_count += 1
                except IndexError:
                    pass
                except Exception as e_proc:
                    print(f"  خطای پردازش box: {e_proc}")
    return total_count, specific_counts

def annotate_frame(frame, total_count, specific_counts, fps=None):
    """نوشتن متن روی فریم."""
    # ... (کد annotate_frame بدون تغییر) ...
    y_offset = 30
    font_scale_large = 0.7
    font_scale_small = 0.6
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_total = (0, 0, 255); color_specific = (0, 255, 0); color_fps = (255, 0, 0)
    bg_color = (0, 0, 0); text_margin = 5

    target_display = 'All Detectable' if TARGET_CLASSES is None else ', '.join(TARGET_CLASSES)
    text_total = f"Total ({target_display}): {total_count}"
    (w_total, h_total), _ = cv2.getTextSize(text_total, font, font_scale_large, thickness)
    cv2.rectangle(frame, (10 - text_margin, y_offset - h_total - text_margin), (10 + w_total + text_margin, y_offset + text_margin), bg_color, -1)
    cv2.putText(frame, text_total, (10, y_offset), font, font_scale_large, color_total, thickness, cv2.LINE_AA)
    y_offset += h_total + 15

    sorted_counts = sorted(specific_counts.items())
    for name, count in sorted_counts:
        text_specific = f"{name}: {count}"
        (w_spec, h_spec), _ = cv2.getTextSize(text_specific, font, font_scale_small, thickness)
        cv2.rectangle(frame, (10 - text_margin, y_offset - h_spec - text_margin), (10 + w_spec + text_margin, y_offset + text_margin), bg_color, -1)
        cv2.putText(frame, text_specific, (10, y_offset), font, font_scale_small, color_specific, thickness, cv2.LINE_AA)
        y_offset += h_spec + 10
        if y_offset > frame.shape[0] - 30: break

    if fps is not None:
        text_fps = f"FPS: {fps:.1f} ({DEVICE.upper()})"
        (w_fps, h_fps), _ = cv2.getTextSize(text_fps, font, font_scale_small, thickness)
        fps_x = frame.shape[1] - w_fps - 10
        cv2.rectangle(frame, (fps_x - text_margin, 30 - h_fps - text_margin), (frame.shape[1] - 10 + text_margin, 30 + text_margin), bg_color, -1)
        cv2.putText(frame, text_fps, (fps_x, 30), font, font_scale_small, color_fps, thickness, cv2.LINE_AA)
    return frame


# --- ترد پردازش ---
def run_yolo_processing(source):
    """تابع اصلی پردازش YOLO در ترد جداگانه."""
    # ... (کد run_yolo_processing با فراخوانی process_predictions و annotate_frame) ...
    global processing_active, model, DEVICE, stop_event, root, current_model_name

    local_model = model
    if not local_model:
        print("خطا داخلی: مدل در دسترس نیست در run_yolo_processing.")
        if root:
            root.after(0, lambda: update_status("خطا: مدل یافت نشد!"))
            root.after(0, _cleanup_after_processing)
        return

    if isinstance(source, int): source_display_name = f'Webcam {source}'
    elif isinstance(source, str): source_display_name = os.path.basename(source)
    else: source_display_name = 'Unknown Source'

    print(f"شروع ترد پردازش برای: {source_display_name} | مدل: {current_model_name} | دستگاه: {DEVICE}")
    model_name_in_status = current_model_name if current_model_name else "Unknown"
    if root:
        root.after(0, lambda: update_status(f"در حال پردازش: {source_display_name} با مدل {model_name_in_status}..."))

    is_image = isinstance(source, str) and source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    is_stream = not is_image
    window_name = f"YOLOv8 Counting - {source_display_name}"
    prev_time = time.time(); frame_count_for_skip = 0; frame_count_for_fps = 0
    accumulated_time = 0; display_fps = 0; active_window = True

    try:
        if is_image:
            window_name += " (Press key/Close)"
            print(f"پردازش تصویر: {source}")
            results_list = local_model.predict(source, stream=False, conf=CONFIDENCE_THRESHOLD, imgsz=INPUT_SIZE, device=DEVICE, verbose=False)
            if results_list and results_list[0].orig_img is not None:
                results = results_list[0]
                annotated_frame = results.plot(conf=True, line_width=2, font_size=None, labels=True)
                total_count, specific_counts = process_predictions(results)
                final_frame = annotate_frame(annotated_frame, total_count, specific_counts)
                cv2.imshow(window_name, final_frame)
                while not stop_event.is_set() and active_window:
                    key = cv2.waitKey(100)
                    if key != -1: break
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: active_window = False; stop_event.set(); break
                    except cv2.error: active_window = False; stop_event.set(); break
            else: print(f"خطا: پردازش تصویر {source_display_name} ناموفق.");# show warning in GUI if needed
        else: # Video/Webcam
            window_name += " (Press 'q'/Close)"
            print(f"پردازش استریم: {source_display_name}")
            results_generator = local_model.predict(source, stream=True, conf=CONFIDENCE_THRESHOLD, imgsz=INPUT_SIZE, device=DEVICE, verbose=False)
            for results in results_generator:
                if stop_event.is_set(): break
                current_time = time.time(); elapsed = current_time - prev_time; prev_time = current_time
                frame_count_for_skip += 1; frame_count_for_fps += 1; accumulated_time += elapsed
                if accumulated_time >= 1.0: display_fps = frame_count_for_fps / accumulated_time; frame_count_for_fps = 0; accumulated_time = 0

                if is_stream and SKIP_FRAMES_FACTOR > 1 and (frame_count_for_skip -1) % SKIP_FRAMES_FACTOR != 0:
                    if active_window:
                        try:
                            key = cv2.waitKey(1)
                            if key != -1 and key & 0xFF == ord('q'): stop_event.set(); break
                            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: active_window = False; stop_event.set(); break
                        except cv2.error: active_window = False; stop_event.set(); break
                        except Exception: active_window = False; stop_event.set(); break
                    if not active_window: break
                    continue

                if not active_window: break
                if results.orig_img is None: continue

                annotated_frame = results.plot(conf=True, line_width=2, font_size=None, labels=True)
                total_count, specific_counts = process_predictions(results)
                final_frame = annotate_frame(annotated_frame, total_count, specific_counts, display_fps)
                try:
                    cv2.imshow(window_name, final_frame)
                    key = cv2.waitKey(1)
                    if key != -1 and key & 0xFF == ord('q'): stop_event.set(); break
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: active_window = False; stop_event.set(); break
                except cv2.error: active_window = False; stop_event.set(); break
                except Exception: active_window = False; stop_event.set(); break
            print(f"پردازش استریم {source_display_name} کامل شد/متوقف شد.")
    except FileNotFoundError as e:
         if root: root.after(0, lambda err=e: messagebox.showerror("خطای فایل", f"فایل منبع یافت نشد:\n{source}\n\n{err}", parent=root))
    except cv2.error as e:
         if root:
             err_copy = e
             # ... (Show specific OpenCV errors) ...
             root.after(0, lambda err=err_copy: messagebox.showerror("خطای OpenCV", f"خطای OpenCV:\n{err}", parent=root))
    except Exception as e:
        print(f"خطای غیرمنتظره در ترد پردازش ({source_display_name}): {e}")
        traceback.print_exc()
        if root: root.after(0, lambda err=e: messagebox.showerror("خطای پردازش", f"خطای پردازش:\n{err}\n\nکنسول را ببینید.", parent=root))
    finally:
        print(f"پایان ترد پردازش برای {source_display_name}.")
        if active_window:
            try:
                 if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow(window_name)
            except: pass # Ignore errors destroying window
        if root:
            root.after(0, _cleanup_after_processing)


def _cleanup_after_processing():
    """تمیزکاری پس از پردازش."""
    global processing_active
    if not root: return
    if processing_active:
        processing_active = False
        stop_event.clear()
        enable_controls()
        model_str = current_model_name if current_model_name else "هیچ مدلی"
        update_status(f"آماده. (مدل: {model_str}, دستگاه: {DEVICE.upper()})")
        print("تمیزکاری کامل شد.")


# --- توابع مربوط به GUI ---

def select_image():
    if not root or processing_active: return
    if not model: messagebox.showerror("خطا", "لطفاً ابتدا یک مدل بارگذاری کنید.", parent=root); return
    filepath = filedialog.askopenfilename(title="انتخاب عکس", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")], parent=root)
    if filepath: start_processing(filepath)

def select_video():
    if not root or processing_active: return
    if not model: messagebox.showerror("خطا", "لطفاً ابتدا یک مدل بارگذاری کنید.", parent=root); return
    filepath = filedialog.askopenfilename(title="انتخاب ویدیو", filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")], parent=root)
    if filepath: start_processing(filepath)

def use_webcam():
    if not root or processing_active: return
    if not model: messagebox.showerror("خطا", "لطفاً ابتدا یک مدل بارگذاری کنید.", parent=root); return
    webcam_id = 0; cap = None
    try:
        cap = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW) # Try DSHOW first
        if not cap or not cap.isOpened():
            if cap: cap.release()
            cap = cv2.VideoCapture(webcam_id)
            if not cap or not cap.isOpened(): messagebox.showerror("خطای وب‌کم", f"وب‌کم ({webcam_id}) یافت نشد.", parent=root); return
        cap.release(); cap = None
        start_processing(webcam_id)
    except Exception as e: messagebox.showerror("خطای وب‌کم", f"خطا در دسترسی وب‌کم:\n{e}", parent=root)
    finally:
        if cap and cap.isOpened(): cap.release()


def start_processing(source):
    """شروع ترد پردازش."""
    global processing_active, yolo_thread, stop_event
    if not root or processing_active or not model: return
    processing_active = True
    stop_event.clear()
    disable_controls()
    print("ایجاد و شروع ترد پردازش YOLO...")
    yolo_thread = threading.Thread(target=run_yolo_processing, args=(source,), daemon=True)
    yolo_thread.start()

def stop_processing():
    """ارسال سیگنال توقف."""
    global processing_active, yolo_thread
    if not root or not processing_active: return
    if yolo_thread is not None and yolo_thread.is_alive():
        print("درخواست توقف پردازش...")
        root.after(0, lambda: update_status("در حال متوقف کردن..."))
        stop_event.set()
        safely_configure_widget(btn_stop, state=tk.DISABLED)
    else:
         _cleanup_after_processing()


def safely_configure_widget(widget, **options):
    """پیکربندی امن ویجت."""
    if widget and isinstance(widget, (tk.Widget, ttk.Widget)):
        try:
            if widget.winfo_exists(): widget.config(**options)
        except tk.TclError: pass

def disable_processing_buttons():
    """غیرفعال کردن دکمه‌های شروع پردازش."""
    safely_configure_widget(btn_image, state=tk.DISABLED)
    safely_configure_widget(btn_video, state=tk.DISABLED)
    safely_configure_widget(btn_webcam, state=tk.DISABLED)

def disable_controls_for_loading():
    """غیرفعال کردن کنترل‌ها هنگام بارگذاری مدل."""
    disable_processing_buttons()
    safely_configure_widget(model_select_combo, state=tk.DISABLED)
    safely_configure_widget(btn_load_model, state=tk.DISABLED)
    safely_configure_widget(btn_stop, state=tk.DISABLED)

def disable_controls():
    """غیرفعال کردن کنترل‌ها هنگام پردازش."""
    disable_controls_for_loading() # هنگام پردازش، بارگذاری مدل هم غیرفعال است
    safely_configure_widget(btn_stop, state=tk.NORMAL) # دکمه توقف فعال می‌شود

def enable_controls():
    """فعال کردن کنترل‌ها پس از اتمام کار."""
    if not root: return
    safely_configure_widget(model_select_combo, state='readonly') # یا tk.NORMAL اگر تایپ مجاز باشد
    safely_configure_widget(btn_load_model, state=tk.NORMAL)
    safely_configure_widget(btn_stop, state=tk.DISABLED)
    model_ready = (model is not None)
    new_state = tk.NORMAL if model_ready else tk.DISABLED
    safely_configure_widget(btn_image, state=new_state)
    safely_configure_widget(btn_video, state=new_state)
    safely_configure_widget(btn_webcam, state=new_state)


def update_status(message):
    """به‌روزرسانی برچسب وضعیت."""
    if status_label and status_label.winfo_exists():
         safely_configure_widget(status_label, text=f"وضعیت: {message}")


def on_closing():
    """هنگام بستن پنجره."""
    global processing_active, root, yolo_thread, model, DEVICE
    if not root: return
    print("درخواست بستن پنجره...")
    close_confirmed = True
    if processing_active and yolo_thread is not None and yolo_thread.is_alive():
        response = messagebox.askyesno("تایید خروج", "پردازش فعال است. خارج می‌شوید؟", icon='warning', parent=root)
        if response:
            print("توقف پردازش برای خروج...")
            stop_event.set()
            update_status("در حال توقف برای خروج...")
            yolo_thread.join(timeout=1.5)
            if yolo_thread.is_alive(): print("هشدار: ترد پردازش هنوز زنده است.")
            processing_active = False
        else: print("خروج لغو شد."); close_confirmed = False

    if close_confirmed:
        print("بستن پنجره و آزادسازی منابع...")
        cv2.destroyAllWindows()
        if model is not None and DEVICE == 'cuda':
             print("آزادسازی مدل از CUDA...")
             try: del model; torch.cuda.empty_cache(); print("CUDA آزاد شد.")
             except Exception as e: print(f"خطا در آزادسازی نهایی مدل: {e}")
        print("نابود کردن root..."); root.destroy(); root = None; print("Root نابود شد.")


# =====================================================
#   بخش اصلی اجرای برنامه
# =====================================================

if __name__ == "__main__":
    print("="*60 + "\n      شمارشگر اشیاء با YOLOv8 و Tkinter\n" + "="*60)
    print(f"PyTorch: {torch.__version__}, Models: {AVAILABLE_MODELS}, Default: {DEFAULT_MODEL_NAME}")
    print(f"Conf: {CONFIDENCE_THRESHOLD}, Input: {INPUT_SIZE}, Skip: {SKIP_FRAMES_FACTOR}")
    print("-"*60)

    print("1. بررسی دستگاه (CUDA/CPU)...")
    check_cuda()
    print("-"*60)

    print("2. ایجاد GUI پایه...")
    gui_init_successful = False
    try:
        root = tk.Tk()
        root.title("شمارشگر اشیاء YOLOv8")
        root.geometry("520x420") # کمی عریض‌تر/بلندتر
        root.minsize(500, 400)

        style = ttk.Style()
        try: style.theme_use('vista') # یا 'xpnative', 'clam'
        except tk.TclError: print("تم 'vista' یافت نشد، استفاده از پیش‌فرض.")

        main_frame = ttk.Frame(root, padding="15 10 15 15") # کمی پدینگ کمتر
        main_frame.pack(expand=True, fill=tk.BOTH)
        main_frame.columnconfigure(0, weight=1)
        # main_frame.rowconfigure(2, weight=1) # ردیف دکمه توقف وزن نگیرد

        # --- ویجت‌های مدل (قبل از بارگذاری) ---
        model_frame = ttk.LabelFrame(main_frame, text=" ۱. انتخاب و بارگذاری مدل ", padding="10 5 10 10")
        model_frame.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        # ستون‌ها: 0:Label, 1:Combobox(weight=1), 2:Button
        model_frame.columnconfigure(1, weight=1)

        # Combobox برای انتخاب مدل
        model_select_label = ttk.Label(model_frame, text="انتخاب مدل:")
        model_select_label.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")
        model_select_combo = ttk.Combobox(model_frame, values=AVAILABLE_MODELS, state='readonly', width=10)
        model_select_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        model_select_combo.set(DEFAULT_MODEL_NAME) # تنظیم پیش‌فرض

        # دکمه بارگذاری مدل انتخاب شده
        btn_load_model = ttk.Button(model_frame, text="بارگذاری", command=load_selected_model_from_combobox, width=10)
        btn_load_model.grid(row=0, column=2, padx=(5, 0), pady=5)

        # Entry برای نمایش مدل بارگذاری شده
        model_display_label = ttk.Label(model_frame, text="مدل فعلی:")
        model_display_label.grid(row=1, column=0, padx=(0, 5), pady=(5,0), sticky="w")
        model_display_entry = ttk.Entry(model_frame, state='readonly')
        model_display_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=(5,0), sticky="ew") # تمام عرض باقیمانده

        # برچسب وضعیت (قبل از بارگذاری)
        status_label = ttk.Label(main_frame, text="وضعیت: آماده برای بارگذاری مدل...", relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_label.grid(row=3, column=0, sticky="ew", pady=(10, 0), ipady=3)

        root.update_idletasks() # نمایش ویجت‌ها قبل از بارگذاری

        # 3. تلاش برای بارگذاری مدل پیش‌فرض
        print(f"3. تلاش برای بارگذاری مدل پیش‌فرض '{DEFAULT_MODEL_NAME}.pt'...")
        # غیرفعال کردن کنترل‌ها هنگام بارگذاری اولیه
        disable_controls_for_loading()
        initial_model_loaded = load_yolo_model(DEFAULT_MODEL_NAME + '.pt')
        if initial_model_loaded: print(f"مدل پیش‌فرض بارگذاری شد.")
        else: print(f"هشدار: بارگذاری مدل پیش‌فرض ناموفق.")
        # enable_controls توسط finally در load_yolo_model صدا زده می‌شود
        print("-"*60)

        # 4. ایجاد بقیه ویجت‌های GUI
        print("4. ایجاد بقیه ویجت‌ها...")

        # فریم انتخاب منبع
        source_frame = ttk.LabelFrame(main_frame, text=" ۲. انتخاب منبع پردازش ", padding="10 10 10 10")
        source_frame.grid(row=1, column=0, pady=(0,10), sticky="ew")
        source_frame.columnconfigure(0, weight=1)
        source_frame.columnconfigure(1, weight=1)

        btn_image = ttk.Button(source_frame, text="انتخاب عکس", command=select_image)
        btn_image.grid(row=0, column=0, padx=5, pady=5, ipady=4, sticky="ew")
        btn_video = ttk.Button(source_frame, text="انتخاب ویدیو", command=select_video)
        btn_video.grid(row=0, column=1, padx=5, pady=5, ipady=4, sticky="ew")
        btn_webcam = ttk.Button(source_frame, text="استفاده از وب‌کم", command=use_webcam)
        btn_webcam.grid(row=1, column=0, columnspan=2, padx=5, pady=5, ipady=4, sticky="ew")

        # دکمه توقف
        btn_stop = ttk.Button(main_frame, text="توقف پردازش", command=stop_processing, state=tk.DISABLED)
        btn_stop.grid(row=2, column=0, pady=(5, 10), ipady=5, sticky="ew", padx=10)

        # 5. تنظیمات نهایی
        root.protocol("WM_DELETE_WINDOW", on_closing)
        print("تنظیم وضعیت نهایی کنترل‌ها...")
        # enable_controls() # اطمینان از وضعیت صحیح پس از بارگذاری اولیه
        # وضعیت اولیه در finally بارگذاری مدل تنظیم شده است. وضعیت فعلی کافی است.
        if not model: update_status(f"آماده. مدل پیش‌فرض ({DEFAULT_MODEL_NAME}) بارگذاری نشد. لطفاً مدلی را انتخاب و بارگذاری کنید.")

        # 6. شروع حلقه اصلی
        print("رابط کاربری آماده. شروع حلقه اصلی Tkinter...")
        print("-"*60)
        gui_init_successful = True
        root.mainloop()

        # پس از بسته شدن پنجره
        print("="*60 + "\nحلقه Tkinter پایان یافت. برنامه بسته شد.\n" + "="*60)

    except Exception as gui_error:
        print("\n" + "="*60 + "\nخطای بحرانی GUI:\n")
        traceback.print_exc()
        print("="*60)
        try: # تلاش برای نمایش پیام خطا
            temp_root = tk.Tk(); temp_root.withdraw()
            messagebox.showerror("خطای GUI", f"خطای GUI:\n{gui_error}\n\nبرنامه بسته می‌شود.", parent=None)
            temp_root.after(500, temp_root.destroy); temp_root.mainloop()
        except: pass
