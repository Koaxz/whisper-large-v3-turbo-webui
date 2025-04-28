import torch
from flask import Flask, request, jsonify, render_template_string, send_file
import os
import subprocess
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import uuid
import json
import threading
import shutil # For directory cleanup
import traceback # For detailed error logging

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
TRANSCRIPTION_FOLDER = 'transcriptions'
TEMP_CHUNK_FOLDER = 'temp_chunks'
# Set the path to your ffmpeg executable if it's not in the system PATH
FFMPEG_PATH = 'ffmpeg' # Or specify full path: r'C:\path\to\ffmpeg\bin\ffmpeg.exe'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)
os.makedirs(TEMP_CHUNK_FOLDER, exist_ok=True)

# --- Task Management ---
tasks = {}
tasks_lock = threading.Lock()

# --- Model Initialization ---
model_cache = {}

def get_available_devices():
    """Gets available compute devices (CPU and CUDA GPUs)."""
    devices = [('cpu', 'CPU')]
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                # Basic capability check (won't catch all incompatibilities like the warning)
                try:
                    capability = torch.cuda.get_device_capability(i)
                    devices.append((f'cuda:{i}', f'GPU {i}: {gpu_name} (Compute: {capability[0]}.{capability[1]})'))
                except Exception: # Handle potential errors getting capability
                     devices.append((f'cuda:{i}', f'GPU {i}: {gpu_name} (Capability check failed)'))
    except Exception as e:
        print(f"Error detecting CUDA devices: {e}") # Log error if CUDA detection fails
    return devices

available_devices = get_available_devices()
# Default to first CUDA device if available and seems compatible, else CPU
default_device = 'cpu'
if len(available_devices) > 1:
    # Very basic check: assume cuda:0 might work if detected.
    # User MUST ensure PyTorch compatibility.
    default_device = available_devices[1][0] # Usually 'cuda:0'

# --- Default Language ---
DEFAULT_LANGUAGE = 'ru' # Set default language to Russian

def initialize_model(device_str):
    """Initializes and caches the Whisper model for a specific device."""
    if device_str in model_cache:
        return model_cache[device_str]

    print(f"Initializing model on device: {device_str}")
    try:
        # Determine torch dtype
        if 'cuda' in device_str and torch.cuda.is_available():
            try:
                # Attempt to use float16 for CUDA for better performance
                torch_dtype = torch.float16
                _ = torch.tensor([1.0], dtype=torch_dtype).to(device_str) # Test allocation
                print(f"Using torch_dtype: {torch_dtype} on {device_str}")
            except Exception as e:
                print(f"Warning: Could not use float16 on {device_str}, falling back to float32. Error: {e}")
                torch_dtype = torch.float32
                _ = torch.tensor([1.0], dtype=torch_dtype).to(device_str) # Test float32 allocation
            device = torch.device(device_str)
            low_cpu_mem = True
        else:
            print("Using CPU or CUDA not available/selected.")
            torch_dtype = torch.float32
            device = torch.device('cpu')
            low_cpu_mem = False
            device_str = 'cpu'
            print(f"Using torch_dtype: {torch_dtype} on CPU")

        # Choose your model ID
        # model_id = "openai/whisper-large-v3"
        model_id = "distil-whisper/distil-large-v2" # Smaller, potentially faster

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipeline_device_arg = device if 'cuda' in device_str else 'cpu'

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=pipeline_device_arg,
        )
        model_cache[device_str] = pipe
        print(f"Model initialized successfully for {device_str}.")
        return pipe
    except Exception as e:
        print(f"FATAL: Failed to initialize model on device {device_str}: {e}")
        traceback.print_exc()
        if device_str in model_cache:
            del model_cache[device_str]
        raise RuntimeError(f"Model initialization failed for {device_str}") from e


# --- Transcription Core Logic (Corrected) ---
def process_transcription(audio_path, device_str, language, translate, transcription_id, task_id=None, original_filename="Unknown"):
    """
    Processes a single audio file for transcription.
    Handles model loading, transcription/translation, saving results, and task status updates.
    """
    processing_started = False
    try:
        print(f"[Task {task_id or 'Sync'}] Processing {original_filename} on {device_str}...")

        pipe = initialize_model(device_str)
        processing_started = True

        # Prepare generation arguments
        generate_kwargs = {}

        # --- Language Handling ---
        # If language is provided AND it's not 'auto', use it.
        if language and language != 'auto':
            generate_kwargs['language'] = language
            print(f"[Task {task_id or 'Sync'}] Language explicitly set to: {language}")
        else:
            # If 'auto' or not provided (though routes default to 'ru'), let model detect.
            # No 'language' key is added to generate_kwargs in this case.
            print(f"[Task {task_id or 'Sync'}] Language set to automatic detection by the model.")

        # --- Explicitly set the task (transcribe or translate) --- CORRECTED LOGIC ---
        if translate:
            # Check if translation is permissible (source language is not explicitly English)
            current_transcription_language = generate_kwargs.get('language', None)

            # Allow translation attempt even if language is 'auto'.
            # Forbid translation ONLY if language is explicitly set to 'en'.
            if current_transcription_language != 'en':
                generate_kwargs['task'] = 'translate'
                print(f"[Task {task_id or 'Sync'}] Translation to English requested.")
            else:
                # Source is English, translation makes no sense, force transcribe
                generate_kwargs['task'] = 'transcribe'
                print(f"[Task {task_id or 'Sync'}] Translation skipped (source is English), task set to transcribe.")
        else:
            # If translation is NOT requested, ALWAYS use transcribe
            generate_kwargs['task'] = 'transcribe'
            print(f"[Task {task_id or 'Sync'}] Task explicitly set to transcribe.")
        # --- End of task setting logic ---

        # Perform transcription/translation
        print(f"[Task {task_id or 'Sync'}] Starting inference for {audio_path} with kwargs: {generate_kwargs}...") # Log kwargs
        # Use generate_kwargs argument in the pipeline call
        result = pipe(audio_path, generate_kwargs=generate_kwargs, return_timestamps=False) # Timestamps optional
        transcription_text = result["text"]
        print(f"[Task {task_id or 'Sync'}] Inference completed.")

        # Save transcription/translation to file
        transcription_path = os.path.join(TRANSCRIPTION_FOLDER, f'{transcription_id}.txt')
        os.makedirs(os.path.dirname(transcription_path), exist_ok=True)
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text)
        print(f"[Task {task_id or 'Sync'}] Result saved to {transcription_path}")

        # Update task status if asynchronous
        if task_id:
            with tasks_lock:
                if task_id in tasks:
                    tasks[task_id]['status'] = 'completed'
                    tasks[task_id]['transcription'] = transcription_text
                    tasks[task_id]['id'] = transcription_id
                    print(f"[Task {task_id}] Status updated to completed.")
                else:
                    print(f"[Task {task_id}] Warning: Task ID not found during completion update.")

        return transcription_text

    except Exception as e:
        print(f"[Task {task_id or 'Sync'}] Error during transcription processing: {e}")
        traceback.print_exc()

        if task_id:
            with tasks_lock:
                 if task_id in tasks:
                    tasks[task_id]['status'] = 'error'
                    tasks[task_id]['error'] = str(e)
                    print(f"[Task {task_id}] Status updated to error.")
                 else:
                    print(f"[Task {task_id}] Warning: Task ID not found during error update.")
        else:
            raise e # Re-raise for synchronous calls

    finally:
        # Cleanup intermediate audio file
        if audio_path and os.path.exists(audio_path):
             try:
                 print(f"[Task {task_id or 'Sync'}] Cleaning up intermediate file: {audio_path}")
                 os.remove(audio_path)
             except OSError as err:
                 print(f"[Task {task_id or 'Sync'}] Warning: Could not delete intermediate file {audio_path}: {err}")


def convert_to_wav(input_path, output_path):
    """Converts an input file (audio/video) to WAV format using ffmpeg."""
    try:
        print(f"Converting {input_path} to WAV at {output_path}...")
        command = [
            FFMPEG_PATH, '-y', '-i', input_path,
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', # Enforce Whisper requirements
            output_path
        ]
        # Use capture_output=True for newer Python versions, or stdout/stderr=subprocess.PIPE
        result = subprocess.run(command, check=True, text=True, encoding='utf-8', capture_output=True)
        print("FFmpeg conversion successful.")
        # print("FFmpeg stdout:", result.stdout) # Optional: log ffmpeg output
        # print("FFmpeg stderr:", result.stderr)
        return True
    except FileNotFoundError:
        print(f"Error: '{FFMPEG_PATH}' command not found. Make sure ffmpeg is installed and in your PATH or FFMPEG_PATH is set correctly.")
        raise Exception("FFmpeg not found.")
    except subprocess.CalledProcessError as e:
        print("Error during FFmpeg conversion.")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError: pass
        raise Exception(f"FFmpeg conversion failed: {e.stderr[:500]}")
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")
        traceback.print_exc()
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError: pass
        raise Exception(f"Audio conversion failed: {e}")


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    global available_devices, default_device
    available_devices = get_available_devices()
    if len(available_devices) > 1:
        default_device = available_devices[1][0]
    else:
        default_device = 'cpu'
    return render_template_string(HTML, available_devices=available_devices, default_device=default_device, default_language=DEFAULT_LANGUAGE)

# --- Non-Chunked Transcription ---

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Handles single file uploads (synchronous or asynchronous)."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get parameters, default language to Russian ('ru')
    device_str = request.form.get('device', default_device)
    language = request.form.get('language', DEFAULT_LANGUAGE)
    translate = request.form.get('translate', 'false').lower() == 'true'
    polling = request.form.get('polling', 'false').lower() == 'true'
    original_filename = file.filename
    print(f"Received request for {original_filename}: lang={language}, translate={translate}, polling={polling}, device={device_str}")

    upload_id = str(uuid.uuid4())
    _, file_extension = os.path.splitext(original_filename)
    # Sanitize filename? For now, just use UUID + extension
    saved_filename = f"{upload_id}{file_extension}"
    raw_upload_path = os.path.join(UPLOAD_FOLDER, saved_filename)
    audio_to_process = None # Path to the final WAV file

    try:
        file.save(raw_upload_path)
        print(f"File saved to {raw_upload_path}")

        # Check if conversion is needed
        is_video = file_extension.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv']
        is_common_audio = file_extension.lower() in ['.mp3', '.m4a', '.ogg', '.flac', '.aac', '.opus'] # Added more types
        is_wav = file_extension.lower() == '.wav'

        # Convert if not WAV or if it's video/common audio that might need resampling
        if is_video or is_common_audio or not is_wav:
            wav_filename = f"{upload_id}.wav"
            audio_to_process = os.path.join(UPLOAD_FOLDER, wav_filename)
            if not convert_to_wav(raw_upload_path, audio_to_process):
                 raise Exception("Audio conversion failed.") # convert_to_wav raises on error
            print(f"File converted to WAV: {audio_to_process}")
        else:
            # It's already a WAV, assume it's compatible (16kHz, mono) for now
            # TODO: Could add an ffmpeg check here too to verify WAV properties
            audio_to_process = raw_upload_path
            print(f"File is WAV, using directly: {audio_to_process}")

        transcription_id = str(uuid.uuid4()) # Unique ID for the result file

        if polling:
            # --- Asynchronous ---
            task_id = str(uuid.uuid4())
            with tasks_lock:
                tasks[task_id] = {
                    'status': 'processing',
                    'transcription': None, 'id': None, 'error': None,
                    'filename': original_filename,
                    'start_time': threading.Timer(0, lambda: None) # Placeholder
                }
            print(f"[Task {task_id}] Created for async processing of {original_filename}")

            # Start background thread
            thread = threading.Thread(target=process_transcription,
                                      args=(audio_to_process, device_str, language, translate,
                                            transcription_id, task_id, original_filename))
            thread.daemon = True
            thread.start()

            # Cleanup original upload if conversion happened
            if raw_upload_path != audio_to_process and os.path.exists(raw_upload_path):
                 print(f"[Task {task_id}] Scheduling cleanup for original upload: {raw_upload_path}")
                 # A more robust cleanup might use a queue or finally block in the thread
                 try: os.remove(raw_upload_path)
                 except OSError as e: print(f"Warning: Failed to clean up raw upload {raw_upload_path}: {e}")

            return jsonify({"task_id": task_id}), 202 # Accepted

        else:
            # --- Synchronous ---
            print(f"Starting synchronous processing for {original_filename}...")
            transcription_text = process_transcription(
                audio_path=audio_to_process, # Pass the path to the WAV
                device_str=device_str, language=language, translate=translate,
                transcription_id=transcription_id, task_id=None,
                original_filename=original_filename
            )
            # Cleanup original upload if conversion happened
            if raw_upload_path != audio_to_process and os.path.exists(raw_upload_path):
                try: os.remove(raw_upload_path)
                except OSError as e: print(f"Warning: Failed to clean up raw upload {raw_upload_path}: {e}")

            print(f"Synchronous processing complete for {original_filename}")
            # process_transcription already saved the file.
            return jsonify({"transcription": transcription_text, "id": transcription_id})

    except Exception as e:
        print(f"Error in /transcribe route for {original_filename}: {e}")
        traceback.print_exc()
         # Cleanup any created files on error
        if audio_to_process and os.path.exists(audio_to_process):
            try: os.remove(audio_to_process)
            except OSError: pass
        if raw_upload_path and os.path.exists(raw_upload_path):
            try: os.remove(raw_upload_path)
            except OSError: pass
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


# --- Chunked Transcription ---

@app.route('/transcribe_chunk', methods=['POST'])
def transcribe_chunk():
    """Receives a single chunk of a file upload."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part (chunk)"}), 400
    chunk = request.files['file']
    file_id = request.form.get('fileId')
    chunk_index = request.form.get('chunkIndex')
    total_chunks = request.form.get('totalChunks')

    if not all([file_id, chunk_index is not None, total_chunks is not None]):
        return jsonify({"error": "Missing chunk metadata (fileId, chunkIndex, totalChunks)"}), 400

    try:
        chunk_index = int(chunk_index)
        total_chunks = int(total_chunks)
    except ValueError:
        return jsonify({"error": "Invalid chunk metadata (index/total must be integers)"}), 400

    # Save chunk
    temp_dir = os.path.join(TEMP_CHUNK_FOLDER, file_id)
    os.makedirs(temp_dir, exist_ok=True)
    chunk_filename = os.path.join(temp_dir, f'chunk_{chunk_index:06d}') # Zero-padded for sorting
    try:
        chunk.save(chunk_filename)
        # print(f"Received chunk {chunk_index + 1}/{total_chunks} for fileId {file_id}") # Less verbose logging

        # --- Store parameters with the last chunk ---
        if chunk_index == total_chunks - 1:
            # Default language to Russian ('ru')
            params = {
                'device': request.form.get('device', default_device),
                'language': request.form.get('language', DEFAULT_LANGUAGE),
                'translate': request.form.get('translate', 'false').lower() == 'true',
                'polling': request.form.get('polling', 'false').lower() == 'true',
                'original_filename': request.form.get('originalFilename', f"{file_id}_chunked_file")
            }
            params_path = os.path.join(temp_dir, 'params.json')
            with open(params_path, 'w') as f:
                json.dump(params, f)
            print(f"Parameters saved for fileId {file_id}: {params}")

        # Return minimal success message to reduce log spam
        if (chunk_index + 1) % 10 == 0 or chunk_index == total_chunks - 1: # Log every 10 chunks and the last one
             print(f"Received chunk {chunk_index + 1}/{total_chunks} for fileId {file_id}")

        return jsonify({"message": f"Chunk {chunk_index + 1}/{total_chunks} received"}), 200
    except Exception as e:
        print(f"Error saving chunk {chunk_index} for fileId {file_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to save chunk"}), 500


def reconstruct_file_from_chunks(file_id, temp_dir):
    """Reconstructs the original file from saved chunks."""
    reconstructed_path = None
    try:
        chunk_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('chunk_')])
        if not chunk_files:
            raise ValueError(f"No chunk files found in {temp_dir}.")

        params_path = os.path.join(temp_dir, 'params.json')
        original_filename = f"{file_id}_reconstructed"
        if os.path.exists(params_path):
             with open(params_path, 'r') as f:
                params = json.load(f)
                original_filename = params.get('original_filename', original_filename)

        _, file_extension = os.path.splitext(original_filename)
        if not file_extension: file_extension = ".bin" # Default extension if unknown

        reconstructed_filename = f"{file_id}_reconstructed{file_extension}"
        reconstructed_path = os.path.join(UPLOAD_FOLDER, reconstructed_filename)

        print(f"Reconstructing file {file_id} from {len(chunk_files)} chunks to {reconstructed_path}...")
        with open(reconstructed_path, 'wb') as outfile:
            for chunk_file in chunk_files:
                chunk_path = os.path.join(temp_dir, chunk_file)
                try:
                    with open(chunk_path, 'rb') as infile:
                        outfile.write(infile.read())
                except Exception as read_err:
                    print(f"Error reading chunk file {chunk_path}: {read_err}")
                    raise # Re-raise to stop reconstruction
        print(f"File reconstruction complete: {reconstructed_path}")
        return reconstructed_path, original_filename

    except Exception as e:
        print(f"Error reconstructing file {file_id}: {e}")
        traceback.print_exc()
        if reconstructed_path and os.path.exists(reconstructed_path):
            try: os.remove(reconstructed_path)
            except OSError: pass
        raise


def transcribe_finalize_logic(file_id):
    """Handles reconstruction and transcription initiation for chunked uploads."""
    temp_dir = os.path.join(TEMP_CHUNK_FOLDER, file_id)
    reconstructed_file_path = None
    audio_to_process = None
    original_filename = f"{file_id}_chunked_file" # Default

    if not os.path.isdir(temp_dir):
        print(f"Error: Chunk directory not found for fileId {file_id} at {temp_dir}")
        raise FileNotFoundError("Chunks not found. Upload may be incomplete or expired.")

    try:
        # Load parameters saved with the last chunk
        params_path = os.path.join(temp_dir, 'params.json')
        if not os.path.exists(params_path):
             raise FileNotFoundError(f"Parameters file (params.json) not found in chunk directory {temp_dir}.")
        with open(params_path, 'r') as f:
            params = json.load(f)

        # Use default 'ru' if language missing in params.json for some reason
        device_str = params.get('device', default_device)
        language = params.get('language', DEFAULT_LANGUAGE)
        translate = params.get('translate', False)
        polling = params.get('polling', False)
        original_filename = params.get('original_filename', original_filename)
        print(f"Finalizing {original_filename} (ID: {file_id}): lang={language}, translate={translate}, polling={polling}, device={device_str}")

        # Reconstruct the file
        reconstructed_file_path, _ = reconstruct_file_from_chunks(file_id, temp_dir)

        # --- Conversion Logic (same as non-chunked) ---
        _, file_extension = os.path.splitext(reconstructed_file_path)
        is_video = file_extension.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv']
        is_common_audio = file_extension.lower() in ['.mp3', '.m4a', '.ogg', '.flac', '.aac', '.opus']
        is_wav = file_extension.lower() == '.wav'

        if is_video or is_common_audio or not is_wav:
            wav_filename = f"{file_id}_reconstructed.wav" # Use unique name
            audio_to_process = os.path.join(UPLOAD_FOLDER, wav_filename)
            if not convert_to_wav(reconstructed_file_path, audio_to_process):
                 raise Exception("Audio conversion failed during finalization.")
            print(f"Reconstructed file converted to WAV: {audio_to_process}")
        else:
            audio_to_process = reconstructed_file_path
            print(f"Reconstructed file is WAV, using directly: {audio_to_process}")
        # --- End Conversion Logic ---

        transcription_id = str(uuid.uuid4())

        if polling:
            # --- Asynchronous Finalization ---
            task_id = str(uuid.uuid4())
            with tasks_lock:
                tasks[task_id] = {
                    'status': 'processing',
                    'transcription': None, 'id': None, 'error': None,
                    'filename': original_filename,
                    'start_time': threading.Timer(0, lambda: None)
                }
            print(f"[Task {task_id}] Created for async finalization of chunked file {original_filename} (ID: {file_id})")

            # Pass audio_to_process (the WAV file) to the transcription thread
            thread = threading.Thread(target=process_transcription,
                                      args=(audio_to_process, device_str, language, translate,
                                            transcription_id, task_id, original_filename))
            thread.daemon = True
            thread.start()

            # Cleanup reconstructed (original format) file if conversion happened
            if reconstructed_file_path != audio_to_process and os.path.exists(reconstructed_file_path):
                print(f"[Task {task_id}] Scheduling cleanup for reconstructed file: {reconstructed_file_path}")
                try: os.remove(reconstructed_file_path)
                except OSError as e: print(f"Warning: Failed to clean up reconstructed file {reconstructed_file_path}: {e}")

            # Cleanup temp chunk dir after starting thread
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up chunk directory: {temp_dir}")
            except OSError as e:
                print(f"Warning: Failed to clean up chunk directory {temp_dir}: {e}")

            return {"task_id": task_id}, 202 # Accepted

        else:
            # --- Synchronous Finalization ---
            print(f"Starting synchronous finalization for chunked file {original_filename} (ID: {file_id})...")
            # Pass audio_to_process (the WAV file) to the transcription function
            transcription_text = process_transcription(
                audio_path=audio_to_process,
                device_str=device_str, language=language, translate=translate,
                transcription_id=transcription_id, task_id=None,
                original_filename=original_filename
            )

            # Cleanup reconstructed (original format) file if conversion happened
            if reconstructed_file_path != audio_to_process and os.path.exists(reconstructed_file_path):
                try: os.remove(reconstructed_file_path)
                except OSError as e: print(f"Warning: Failed to clean up reconstructed file {reconstructed_file_path}: {e}")

            # Cleanup temp chunk dir
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up chunk directory: {temp_dir}")
            except OSError as e:
                print(f"Warning: Failed to clean up chunk directory {temp_dir}: {e}")

            print(f"Synchronous finalization complete for {original_filename}")
            return {"transcription": transcription_text, "id": transcription_id}, 200 # OK

    except Exception as e:
        print(f"Error during finalize logic for fileId {file_id}: {e}")
        traceback.print_exc()
         # Attempt broader cleanup on error
        if audio_to_process and os.path.exists(audio_to_process):
            try: os.remove(audio_to_process)
            except OSError: pass
        if reconstructed_file_path and os.path.exists(reconstructed_file_path):
            try: os.remove(reconstructed_file_path)
            except OSError: pass
        if os.path.isdir(temp_dir):
             try: shutil.rmtree(temp_dir)
             except OSError: pass
        # Re-raise the exception so the route returns 500
        raise


@app.route('/transcribe_finalize', methods=['POST'])
def transcribe_finalize():
    """Synchronous finalization endpoint."""
    data = request.get_json()
    if not data or 'fileId' not in data:
        return jsonify({"error": "fileId missing in request body"}), 400
    file_id = data['fileId']
    try:
        result, status_code = transcribe_finalize_logic(file_id)
        return jsonify(result), status_code
    except FileNotFoundError as e:
         return jsonify({"error": str(e)}), 404 # Chunks or params not found
    except Exception as e:
        # Log the error details server-side
        print(f"Error in /transcribe_finalize for fileId {file_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Finalization failed: An internal error occurred."}), 500 # Avoid sending detailed errors

@app.route('/transcribe_finalize_async', methods=['POST'])
def transcribe_finalize_async():
    """Asynchronous finalization endpoint."""
    data = request.get_json()
    if not data or 'fileId' not in data:
        return jsonify({"error": "fileId missing in request body"}), 400
    file_id = data['fileId']
    try:
        # The polling decision is read from params.json inside transcribe_finalize_logic
        result, status_code = transcribe_finalize_logic(file_id)
        # Ensure the result is a task_id if successful async
        if status_code == 202 and 'task_id' not in result:
             print(f"Error: Async finalize logic for {file_id} didn't return task_id correctly.")
             return jsonify({"error": "Internal server error: async task creation failed"}), 500
        return jsonify(result), status_code
    except FileNotFoundError as e:
         return jsonify({"error": str(e)}), 404 # Chunks or params not found
    except Exception as e:
        print(f"Error in /transcribe_finalize_async for fileId {file_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Async finalization failed: An internal error occurred."}), 500


# --- Status and Download ---

@app.route('/status/<task_id>')
def status(task_id):
    """Checks the status of an asynchronous task."""
    # Basic validation of task_id format might be good here (e.g., is it a UUID?)
    try:
        uuid.UUID(task_id)
    except ValueError:
        print(f"Invalid task_id format received: {task_id}")
        return jsonify({"status": "not_found", "error": "Invalid task ID format."}), 400

    with tasks_lock:
        task = tasks.get(task_id)
        if not task:
            # Could optionally check if a result file exists for this ID (if tasks get cleaned up)
            print(f"Task ID not found in active tasks: {task_id}")
            return jsonify({"status": "not_found", "error": "Task ID not found or expired."}), 404

        # Return a copy to avoid race conditions when accessing outside the lock
        task_snapshot = task.copy()

    # Process the snapshot outside the lock
    response = {"status": task_snapshot['status'], "filename": task_snapshot.get('filename', 'Unknown')}
    if task_snapshot['status'] == 'completed':
        response["transcription"] = task_snapshot['transcription']
        response["id"] = task_snapshot['id']
        # Optionally remove completed task from `tasks` dict after some time?
    elif task_snapshot['status'] == 'error':
        # Avoid sending detailed internal errors to the client
        response["error"] = "An error occurred during processing." # Generic error message
        print(f"Task {task_id} failed with error: {task_snapshot.get('error', 'Unknown error')}") # Log detailed error server-side
        # Optionally remove failed task?

    return jsonify(response)


@app.route('/download/<transcription_id>')
def download(transcription_id):
    """Downloads the transcription text file."""
    # Validate transcription_id format and prevent path traversal
    try:
        uuid.UUID(transcription_id) # Validate UUID format
        filename = f"{transcription_id}.txt"
        # Basic path traversal check
        if not filename.isalnum() and '_' not in filename and '.' not in filename:
             # Allow only alphanumeric, underscore, dot - adjust if needed, but be strict
             raise ValueError("Invalid characters in transcription ID.")
        # Securely join path
        transcription_path = os.path.join(TRANSCRIPTION_FOLDER, filename)
        # Check if the resolved path is still within the intended folder
        if not os.path.abspath(transcription_path).startswith(os.path.abspath(TRANSCRIPTION_FOLDER)):
            raise ValueError("Invalid path.")
    except ValueError as e:
        print(f"Invalid download request: {e} (ID: {transcription_id})")
        return jsonify({"error": "Invalid request"}), 400

    if not os.path.exists(transcription_path) or not os.path.isfile(transcription_path):
        print(f"Download request failed: File not found at {transcription_path}")
        return jsonify({"error": "Transcription file not found."}), 404

    try:
        return send_file(transcription_path, as_attachment=True, download_name=f"{transcription_id}_transcription.txt")
    except Exception as e:
        print(f"Error sending file {transcription_path}: {e}")
        traceback.print_exc()
        return jsonify({"error": "Could not send file."}), 500


# --- HTML Template ---
# Includes Russian ('ru') as default, bilingual UI text where possible.
HTML = '''
<!DOCTYPE html>
<html lang="ru"> <!-- Primary language set to Russian -->
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Распознавание речи</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 24px; height: 24px; border-radius: 50%;
            border-left-color: #09f; animation: spin 1s ease infinite;
            display: inline-block; margin-right: 8px; vertical-align: middle;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        #submitButton:disabled { opacity: 0.5; cursor: not-allowed; }
        .result-text { max-height: 300px; overflow-y: auto; background-color: #f9fafb; border: 1px solid #e5e7eb; padding: 8px; border-radius: 4px;} /* Style for text output */
    </style>
</head>
<body class="bg-gray-100 p-8">
    <div class="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-md">
        <h1 class="text-3xl font-bold mb-6 text-center">Whisper Распознавание речи</h1>
        <form id="uploadForm" class="mb-8">
            <!-- Device Selection -->
            <div class="mb-4">
                <label for="deviceSelect" class="block text-sm font-medium text-gray-700 mb-2">Устройство</label>
                <select id="deviceSelect" name="device" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
                    {% for device_id, device_name in available_devices %}
                    <option value="{{ device_id }}" {% if device_id == default_device %}selected{% endif %}>{{ device_name }}</option>
                    {% endfor %}
                </select>
            </div>
             <!-- File Input -->
            <div class="mb-4">
                <label for="fileInput" class="block text-sm font-medium text-gray-700 mb-2">Выберите аудио/видео файл(ы)</label>
                <input type="file" id="fileInput" name="file" accept="audio/*,video/*" multiple required class="block w-full text-sm text-gray-500
                    file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0
                    file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700
                    hover:file:bg-indigo-100 cursor-pointer">
            </div>
             <!-- Language Selection -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                    <label for="languageSelect" class="block text-sm font-medium text-gray-700 mb-2">Язык аудио</label>
                    <select id="languageSelect" name="language" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
                        <option value="ru" {% if default_language == 'ru' %}selected{% endif %}>Русский</option>
                        <option value="auto" {% if default_language == 'auto' %}selected{% endif %}>Автоопределение</option>
                        <option value="en" {% if default_language == 'en' %}selected{% endif %}>English</option>
                        <option value="ja" {% if default_language == 'ja' %}selected{% endif %}>日本語</option>
                        <option value="zh" {% if default_language == 'zh' %}selected{% endif %}>中文</option>
                        <option value="ko" {% if default_language == 'ko' %}selected{% endif %}>한국어</option>
                        <option value="fr" {% if default_language == 'fr' %}selected{% endif %}>Français</option>
                        <option value="de" {% if default_language == 'de' %}selected{% endif %}>Deutsch</option>
                        <option value="es" {% if default_language == 'es' %}selected{% endif %}>Español</option>
                        <!-- Add more common languages -->
                    </select>
                     <p class="mt-1 text-xs text-gray-500">По умолчанию: Русский. "Авто" - модель попытается определить язык.</p>
                </div>
                <!-- Translate Option -->
                 <div>
                    <label for="translateCheck" class="block text-sm font-medium text-gray-700 mb-2">Действие</label>
                    <div class="mt-1 flex items-center h-10"> <!-- Adjust height to match select -->
                        <label class="inline-flex items-center">
                            <input type="checkbox" id="translateCheck" name="translate" value="true" class="form-checkbox h-5 w-5 text-indigo-600 rounded">
                            <span class="ml-2 text-gray-700">Перевести на английский</span>
                        </label>
                    </div>
                     <p class="mt-1 text-xs text-gray-500">Если отмечено, результат будет на английском (если исходный не английский).</p>
                 </div>
             </div>

            <!-- Advanced Options Toggle -->
             <div class="mb-2">
                <button type="button" id="toggleAdvanced" class="text-sm text-indigo-600 hover:text-indigo-800">Расширенные настройки ▼</button>
             </div>

             <!-- Advanced Options Container (Initially Hidden) -->
            <div id="advancedOptions" class="mb-4 border border-gray-200 p-4 rounded-md" style="display: none;">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                     <!-- Chunk Upload -->
                    <div>
                        <label class="inline-flex items-center">
                            <input type="checkbox" id="chunkUploadCheck" class="form-checkbox h-5 w-5 text-indigo-600 rounded" checked>
                            <span class="ml-2 text-gray-700">Загрузка по частям</span>
                        </label>
                         <p class="mt-1 text-xs text-gray-500">Рекомендуется для файлов > 100MB.</p>
                    </div>
                     <!-- Chunk Size -->
                    <div id="chunkSizeContainer">
                        <label for="chunkSizeInput" class="block text-sm font-medium text-gray-700">Размер части (MB)</label>
                        <input type="number" id="chunkSizeInput" min="5" max="100" value="50" class="mt-1 block w-full pl-3 pr-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm">
                         <p class="mt-1 text-xs text-gray-500">5-100MB. Зависит от памяти сервера.</p>
                    </div>
                 </div>
                  <!-- Polling Option -->
                <div class="mt-4">
                    <label class="inline-flex items-center">
                        <input type="checkbox" id="pollingCheck" name="polling" value="true" class="form-checkbox h-5 w-5 text-indigo-600 rounded">
                        <span class="ml-2 text-gray-700">Асинхронный режим (опрос статуса)</span>
                    </label>
                     <p class="mt-1 text-xs text-gray-500">Для очень долгих задач. Результат появится после завершения.</p>
                </div>
             </div>


            <!-- Submit Button -->
            <button type="submit" id="submitButton" class="w-full flex justify-center items-center bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline transition duration-150 ease-in-out">
                <span id="submitButtonText">Начать распознавание</span>
                <div id="submitSpinner" class="spinner" style="display: none;"></div>
            </button>
        </form>
        <!-- Results Area -->
        <h2 class="text-xl font-semibold mb-4 text-center">Результаты</h2>
        <div id="results" class="space-y-4">
             <!-- Result containers will be added here -->
        </div>
    </div>

    <script>
        const uploadForm = document.getElementById('uploadForm');
        const fileInput = document.getElementById('fileInput');
        const deviceSelect = document.getElementById('deviceSelect');
        const languageSelect = document.getElementById('languageSelect');
        const translateCheck = document.getElementById('translateCheck');
        const chunkUploadCheck = document.getElementById('chunkUploadCheck');
        const chunkSizeInput = document.getElementById('chunkSizeInput');
        const pollingCheck = document.getElementById('pollingCheck');
        const resultsDiv = document.getElementById('results');
        const submitButton = document.getElementById('submitButton');
        const submitButtonText = document.getElementById('submitButtonText');
        const submitSpinner = document.getElementById('submitSpinner');
        const chunkSizeContainer = document.getElementById('chunkSizeContainer');
        const toggleAdvanced = document.getElementById('toggleAdvanced');
        const advancedOptions = document.getElementById('advancedOptions');
        const formElements = uploadForm.elements; // Get all form elements once

        // --- Event Listeners ---
        chunkUploadCheck.addEventListener('change', (e) => {
            chunkSizeContainer.style.display = e.target.checked ? 'block' : 'none';
        });

        toggleAdvanced.addEventListener('click', () => {
             const isHidden = advancedOptions.style.display === 'none';
             advancedOptions.style.display = isHidden ? 'block' : 'none';
             toggleAdvanced.textContent = isHidden ? 'Свернуть настройки ▲' : 'Расширенные настройки ▼';
        });


        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            if (fileInput.files.length === 0) {
                alert('Пожалуйста, выберите файл(ы) для загрузки.');
                return;
            }

            setFormDisabled(true);
            resultsDiv.innerHTML = ''; // Clear previous results on new submission

            const useChunkUpload = chunkUploadCheck.checked;
            const chunkSizeMB = parseInt(chunkSizeInput.value, 10) || 50;
            const chunkSize = chunkSizeMB * 1024 * 1024; // Size in Bytes

            const usePolling = pollingCheck.checked;
            const device = deviceSelect.value;
            const language = languageSelect.value; // Read selected language
            const translate = translateCheck.checked; // Read translate checkbox

            // Process each selected file sequentially
            for (let file of fileInput.files) {
                const resultContainerId = `result-${generateUUID()}`;
                const resultDiv = createResultContainer(file.name, resultContainerId);
                resultsDiv.appendChild(resultDiv); // Add container immediately
                updateResultStatus(resultContainerId, file.name, "Подготовка...");

                try {
                    if (usePolling) {
                        // --- Asynchronous Path ---
                        let taskId;
                        updateResultStatus(resultContainerId, file.name, "Загрузка и постановка в очередь...", true);
                        if (useChunkUpload && file.size > chunkSize) { // Only use chunks if file is larger than chunk size
                            const finalResponse = await uploadFileInChunks(file, chunkSize, device, language, translate, true, file.name, resultContainerId);
                            taskId = finalResponse.task_id;
                        } else {
                            const response = await uploadFileStandard(file, device, language, translate, true, resultContainerId);
                            taskId = response.task_id;
                        }
                        updateResultStatus(resultContainerId, file.name, `В очереди (ID Задачи: ${taskId}). Ожидание обработки...`, true);
                        pollTranscription(taskId, file.name, resultContainerId); // Start polling for this task
                    } else {
                        // --- Synchronous Path ---
                        let transcriptionData;
                        if (useChunkUpload && file.size > chunkSize) {
                             updateResultStatus(resultContainerId, file.name, "Загрузка по частям...", true);
                            transcriptionData = await uploadFileInChunks(file, chunkSize, device, language, translate, false, file.name, resultContainerId);
                            updateResultStatus(resultContainerId, file.name, "Обработка файла...", true); // Status before final result
                        } else {
                            updateResultStatus(resultContainerId, file.name, "Загрузка файла...", true);
                            transcriptionData = await uploadFileStandard(file, device, language, translate, false, resultContainerId);
                            updateResultStatus(resultContainerId, file.name, "Обработка файла...", true);
                        }
                        displayResult(resultContainerId, file.name, transcriptionData); // Display final result
                    }
                } catch (error) {
                    console.error("Ошибка обработки файла:", file.name, error);
                    // Display error in the specific container for this file
                    displayError(resultContainerId, file.name, error.message || 'Произошла неизвестная ошибка.');
                }
            } // End loop through files

            setFormDisabled(false); // Re-enable form after attempting all files
        });

        // --- UI Update Functions ---

        function setFormDisabled(disabled) {
            // Disable/enable all form elements to prevent changes during processing
            for (let element of formElements) {
                // Don't disable the results div or buttons inside it
                if (!resultsDiv.contains(element)) {
                     element.disabled = disabled;
                }
            }
            submitSpinner.style.display = disabled ? 'inline-block' : 'none';
            submitButtonText.textContent = disabled ? 'Обработка...' : 'Начать распознавание';
            // Make sure advanced options toggle reflects disabled state visually if needed
            toggleAdvanced.style.opacity = disabled ? 0.5 : 1;
            toggleAdvanced.style.cursor = disabled ? 'not-allowed' : 'pointer';
        }

        function createResultContainer(filename, containerId) {
            const div = document.createElement('div');
            div.id = containerId;
            div.className = 'bg-gray-50 p-4 rounded-lg shadow border border-gray-200'; // Added border
            div.innerHTML = `
                <h3 class="font-semibold mb-2 text-gray-800 break-all">${filename}</h3>
                <div class="status-message text-gray-600 mb-2">
                    <div class="spinner" style="display: none; width: 16px; height: 16px;"></div>
                    <span class="ml-1">Инициализация...</span>
                </div>
                <div class="progress-bar-container mt-1 mb-2" style="display: none;">
                    <div class="w-full bg-gray-200 rounded-full h-1.5">
                        <div class="progress-bar bg-indigo-600 h-1.5 rounded-full" style="width: 0%"></div>
                    </div>
                    <span class="progress-text text-xs text-gray-500 ml-1">0%</span>
                </div>
                <div class="result-content mt-2" style="display: none;">
                    <p class="result-text mb-2 whitespace-pre-wrap text-sm"></p> <!-- Added result-text class -->
                    <div class="action-buttons mt-2">
                        <button onclick="copyToClipboard(this)" class="bg-green-100 hover:bg-green-200 text-green-800 font-semibold py-1 px-2 text-xs rounded mr-2 transition duration-150 ease-in-out">
                            Копировать
                        </button>
                        <a href="#" class="download-link bg-blue-100 hover:bg-blue-200 text-blue-800 font-semibold py-1 px-2 text-xs rounded transition duration-150 ease-in-out" style="display:none;">
                            Скачать (.txt)
                        </a>
                    </div>
                </div>
                <div class="error-message text-red-600 mt-2 text-sm" style="display: none;"></div>
            `;
            return div;
        }

        function updateResultStatus(containerId, filename, message, showSpinner = false) {
            const container = document.getElementById(containerId);
            if (!container) return;
            const statusDiv = container.querySelector('.status-message');
            const spinner = statusDiv.querySelector('.spinner');
            const span = statusDiv.querySelector('span');
            spinner.style.display = showSpinner ? 'inline-block' : 'none';
            span.textContent = message;
            statusDiv.style.display = 'block';
            // Hide other parts when status updates
            container.querySelector('.result-content').style.display = 'none';
            container.querySelector('.error-message').style.display = 'none';
             // Keep progress bar if it was visible? Or hide? Let's hide on general status update.
            // container.querySelector('.progress-bar-container').style.display = 'none';
        }

         function updateUploadProgress(containerId, percentage) {
            const container = document.getElementById(containerId);
            if (!container) return;
            const progressBarContainer = container.querySelector('.progress-bar-container');
            const progressBar = container.querySelector('.progress-bar');
            const progressText = container.querySelector('.progress-text');

            progressBarContainer.style.display = 'block';
            const clampedPercentage = Math.max(0, Math.min(100, Math.round(percentage)));
            progressBar.style.width = `${clampedPercentage}%`;
            progressText.textContent = `${clampedPercentage}%`;

            // Optionally update status message during progress
            // updateResultStatus(containerId, null, `Загрузка... ${clampedPercentage}%`, true);
        }

        function displayResult(containerId, filename, data) {
            const container = document.getElementById(containerId);
            if (!container) return;
            container.querySelector('.status-message').style.display = 'none';
            container.querySelector('.error-message').style.display = 'none';
            container.querySelector('.progress-bar-container').style.display = 'none';

            const resultContent = container.querySelector('.result-content');
            const textParagraph = resultContent.querySelector('p.result-text'); // Target the styled paragraph
            textParagraph.textContent = data.transcription || '（Результат пуст）'; // Display result or empty message
            const downloadLink = resultContent.querySelector('.download-link');

            if (data.id) {
                 downloadLink.href = `/download/${data.id}`;
                 downloadLink.style.display = 'inline-block';
            } else {
                 downloadLink.style.display = 'none';
            }
            resultContent.style.display = 'block'; // Show the results section
        }

        function displayError(containerId, filename, errorMsg) {
             const container = document.getElementById(containerId);
            if (!container) return;
            // Hide other elements
            container.querySelector('.status-message').style.display = 'none';
            container.querySelector('.result-content').style.display = 'none';
            container.querySelector('.progress-bar-container').style.display = 'none';

            const errorDiv = container.querySelector('.error-message');
            // Show a user-friendly error message
            errorDiv.textContent = `Ошибка: ${errorMsg.includes('Failed to fetch') ? 'Сетевая ошибка или сервер недоступен.' : errorMsg}`;
            errorDiv.style.display = 'block';
        }

        // --- Upload Functions ---

        async function uploadFileStandard(file, device, language, translate, polling, containerId) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('device', device);
            formData.append('language', language);
            formData.append('translate', translate);
            formData.append('polling', polling);

            // Use XMLHttpRequest to track upload progress
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open('POST', '/transcribe', true); // Target the single transcribe endpoint

                xhr.upload.onprogress = (event) => {
                    if (event.lengthComputable) {
                        const percentComplete = (event.loaded / event.total) * 100;
                        updateUploadProgress(containerId, percentComplete);
                        // Can refine status message based on progress
                        // if (percentComplete < 100) {
                        //      updateResultStatus(containerId, file.name, `Загрузка... ${Math.round(percentComplete)}%`, true);
                        // }
                    }
                };

                 xhr.onloadstart = () => {
                    updateResultStatus(containerId, file.name, `Загрузка файла...`, true);
                };

                xhr.onload = () => {
                     if (xhr.status >= 200 && xhr.status < 300) {
                        updateResultStatus(containerId, file.name, `Файл загружен, обработка...`, true);
                        try {
                            resolve(JSON.parse(xhr.responseText));
                        } catch (e) {
                             console.error("Invalid JSON response:", xhr.responseText);
                             reject(new Error("Неверный ответ от сервера."));
                        }
                    } else {
                        // Attempt to parse error from server response
                        let errorMsg = `Ошибка сервера (${xhr.status})`;
                        try {
                            const errorData = JSON.parse(xhr.responseText);
                            errorMsg = errorData.error || errorMsg;
                        } catch (e) { /* Ignore parse error, use status text */ }
                         console.error("Upload failed:", xhr.status, xhr.statusText, xhr.responseText);
                        reject(new Error(errorMsg));
                    }
                };

                xhr.onerror = () => {
                    console.error("Network error during upload");
                    reject(new Error('Сетевая ошибка при загрузке файла.'));
                };

                 xhr.ontimeout = () => {
                     console.error("Upload timed out");
                    reject(new Error('Время ожидания загрузки истекло.'));
                };

                xhr.send(formData);
            });
        }


        async function uploadFileInChunks(file, chunkSize, device, language, translate, polling, originalFilename, containerId) {
            const totalChunks = Math.ceil(file.size / chunkSize);
            const fileId = generateUUID();
            let chunksUploaded = 0;

            console.log(`Начало загрузки ${originalFilename} (${(file.size / 1024 / 1024).toFixed(2)} MB) по частям (${totalChunks} шт.) ID: ${fileId}`);
            updateResultStatus(containerId, originalFilename, `Загрузка части 1/${totalChunks}...`, true);

            for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
                const start = chunkIndex * chunkSize;
                const end = Math.min(start + chunkSize, file.size);
                const chunk = file.slice(start, end);

                const formData = new FormData();
                formData.append('file', chunk, `${originalFilename}.part${chunkIndex}`); // Add filename to chunk for clarity
                formData.append('fileId', fileId);
                formData.append('chunkIndex', chunkIndex);
                formData.append('totalChunks', totalChunks);
                // Include all necessary parameters with the *last* chunk (backend reads params.json)
                // Sending with every chunk is slightly redundant but harmless if backend handles it correctly.
                // Crucially, the *last* chunk *must* have them for params.json creation.
                if (chunkIndex === totalChunks - 1) {
                    formData.append('device', device);
                    formData.append('language', language);
                    formData.append('translate', translate);
                    formData.append('polling', polling);
                    formData.append('originalFilename', originalFilename);
                 }

                const chunkEndpoint = '/transcribe_chunk'; // Single endpoint for chunks

                try {
                    // Use fetch, progress is tracked by chunk count
                    const response = await fetch(chunkEndpoint, { method: 'POST', body: formData });

                    if (!response.ok) {
                        let errorMsg = `Ошибка загрузки части ${chunkIndex + 1} (${response.status})`;
                         try {
                            const errorData = await response.json();
                            errorMsg = errorData.error || errorMsg;
                        } catch (e) { /* Ignore parse error */ }
                        throw new Error(errorMsg);
                    }

                    chunksUploaded++;
                    const percentComplete = (chunksUploaded / totalChunks) * 100;
                    updateUploadProgress(containerId, percentComplete); // Update progress bar
                     if (chunksUploaded < totalChunks) {
                        updateResultStatus(containerId, originalFilename, `Загрузка части ${chunksUploaded + 1}/${totalChunks}...`, true);
                    } else {
                         updateResultStatus(containerId, originalFilename, `Все части загружены (${totalChunks}/${totalChunks}). Сборка файла...`, true);
                    }

                    // Optional small delay between chunks? Probably not necessary.
                    // await new Promise(resolve => setTimeout(resolve, 10));

                } catch (error) {
                    console.error(`Ошибка загрузки части ${chunkIndex + 1}: ${error.message}`);
                    // Throw error to stop the loop for this file
                    throw new Error(`Ошибка при загрузке части ${chunkIndex + 1}: ${error.message}`);
                }
            } // End chunk loop

            console.log(`Все части загружены для ${fileId}. Запрос на финализацию...`);

            // Finalize the upload (reconstruct and start processing)
            const finalizeEndpoint = polling ? '/transcribe_finalize_async' : '/transcribe_finalize';
            const finalResponse = await fetch(finalizeEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fileId: fileId }) // Only need fileId
            });

            if (!finalResponse.ok) {
                let errorMsg = `Ошибка финализации (${finalResponse.status})`;
                 try {
                    const errorData = await finalResponse.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) { /* Ignore */ }
                 console.error(`Finalization failed for ${fileId}: ${errorMsg}`);
                throw new Error(errorMsg);
            }

             console.log(`Финализация для ${fileId} запрошена успешно.`);
             // Return the response from finalize (contains task_id or transcription)
             return await finalResponse.json();
        }


        // --- Polling Function ---

        async function pollTranscription(taskId, filename, containerId) {
            console.log(`Начало опроса статуса для задачи ${taskId} (${filename})`);
            // Initial status update handled before calling poll

            const pollInterval = 5000; // Check every 5 seconds
            let consecutiveErrors = 0;
            const maxErrors = 6; // Increase max errors slightly

            const intervalId = setInterval(async () => {
                try {
                    const response = await fetch(`/status/${taskId}`);

                    // Handle network errors or server down
                    if (!response.ok) {
                        if (response.status === 404) {
                            console.warn(`Задача ${taskId} не найдена сервером (возможно, завершена и очищена или неверный ID).`);
                             throw new Error(`Задача ${taskId} не найдена.`); // Stop polling
                        }
                         // Other server errors (5xx)
                         throw new Error(`Ошибка сервера при проверке статуса (${response.status}).`);
                    }

                    const data = await response.json();
                    consecutiveErrors = 0; // Reset error count on successful fetch

                    // Update UI based on status
                    switch(data.status) {
                        case 'completed':
                            clearInterval(intervalId);
                            console.log(`Задача ${taskId} (${filename}) завершена.`);
                            displayResult(containerId, data.filename || filename, data);
                            break;
                        case 'error':
                            clearInterval(intervalId);
                            console.error(`Задача ${taskId} (${filename}) завершилась с ошибкой: ${data.error}`);
                            displayError(containerId, data.filename || filename, data.error || 'Произошла ошибка обработки.');
                            break;
                        case 'processing':
                            console.log(`Задача ${taskId} (${filename}) все еще обрабатывается...`);
                            // More specific status message? Could potentially get progress info if backend provided it.
                            updateResultStatus(containerId, data.filename || filename, 'Идет обработка...', true);
                            break;
                        case 'not_found': // Should be caught by !response.ok, but handle defensively
                             clearInterval(intervalId);
                             console.warn(`Задача ${taskId} не найдена (ответ от API).`);
                             displayError(containerId, filename, `Задача ${taskId} не найдена.`);
                            break;
                        default:
                            // Unknown status - log it, maybe stop polling?
                             console.warn(`Неизвестный статус задачи ${taskId}: ${data.status}`);
                             updateResultStatus(containerId, data.filename || filename, `Неизвестный статус: ${data.status}`, true);
                             // Possibly stop polling if status is unexpected
                             // clearInterval(intervalId);
                    }

                } catch (error) {
                    console.error(`Ошибка опроса для задачи ${taskId}: ${error.message}`);
                    consecutiveErrors++;
                    updateResultStatus(containerId, filename, `Ошибка опроса (${consecutiveErrors}/${maxErrors})...`, true);

                    if (consecutiveErrors >= maxErrors) {
                         clearInterval(intervalId);
                         console.error(`Превышено максимальное количество ошибок опроса для задачи ${taskId}. Остановка.`);
                         displayError(containerId, filename, `Не удалось получить статус задачи после нескольких попыток: ${error.message}.`);
                    }
                    // Continue polling after transient errors
                }
            }, pollInterval);
        }


        // --- Utilities ---

        function generateUUID() { // Basic UUID generator
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }

        function copyToClipboard(button) {
            // Find the text paragraph within the same result-content container
            const resultContainer = button.closest('.result-content');
            if (!resultContainer) {
                console.error("Could not find result container for copy button.");
                return;
            }
            const textToCopy = resultContainer.querySelector('p.result-text')?.textContent; // Target the styled paragraph

            if (textToCopy === null || textToCopy === undefined || textToCopy.trim() === '') {
                 alert('Нет текста для копирования.');
                return;
            }

            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalText = button.textContent;
                button.textContent = 'Скопировано!';
                button.disabled = true;
                // Briefly change style to indicate success
                 button.classList.remove('bg-green-100', 'hover:bg-green-200', 'text-green-800');
                 button.classList.add('bg-green-500', 'text-white');
                setTimeout(() => {
                    button.textContent = originalText;
                    button.disabled = false;
                    button.classList.add('bg-green-100', 'hover:bg-green-200', 'text-green-800');
                    button.classList.remove('bg-green-500', 'text-white');
                }, 2000); // Revert after 2 seconds
            }).catch(err => {
                console.error('Ошибка копирования в буфер обмена: ', err);
                alert('Не удалось скопировать текст. Ваш браузер может не поддерживать эту функцию или требовать HTTPS.');
            });
        }

    </script>
</body>
</html>
'''

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Whisper Flask Server ---")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Transcription folder: {os.path.abspath(TRANSCRIPTION_FOLDER)}")
    print(f"Temp chunk folder: {os.path.abspath(TEMP_CHUNK_FOLDER)}")
    print(f"FFmpeg path: {FFMPEG_PATH}")
    print(f"Available devices: {available_devices}")
    print(f"Default device: {default_device}")
    print(f"Default language: {DEFAULT_LANGUAGE}")
    print("-" * 30)
    print("Starting server on http://0.0.0.0:5000 ...")
    print("IMPORTANT: If using GPU, ensure PyTorch is correctly installed with CUDA support matching your drivers and GPU capability!")
    print("-" * 30)

    # Use waitress or gunicorn in production for better performance and stability
    # Example using waitress (install with: pip install waitress)
    try:
        from waitress import serve
        print("Using Waitress server.")
        serve(app, host='0.0.0.0', port=5000, threads=8) # Adjust threads as needed
    except ImportError:
        print("Waitress not found, falling back to Flask development server (not recommended for production).")
        print("Install waitress: pip install waitress")
        # Development server (use only for testing)
        # debug=True enables auto-reloading but can cause issues with models and threads
        app.run(host='0.0.0.0', port=5000, debug=False)