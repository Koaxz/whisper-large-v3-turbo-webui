import torch
from flask import Flask, request, jsonify, render_template_string, send_file
import os
import subprocess
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import uuid
import json
import threading
import shutil # For directory cleanup

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
                # Test allocation to catch potential CUDA errors early
                _ = torch.tensor([1.0], dtype=torch_dtype).to(device_str)
                print(f"Using torch_dtype: {torch_dtype} on {device_str}")
            except Exception as e:
                print(f"Warning: Could not use float16 on {device_str}, falling back to float32. Error: {e}")
                torch_dtype = torch.float32
                # Test float32 allocation
                _ = torch.tensor([1.0], dtype=torch_dtype).to(device_str)
            device = torch.device(device_str)
            low_cpu_mem = True # Keep True if loading on GPU
            # Check PyTorch compatibility warning again here if possible
            # Note: The warning user saw happens during import/initial checks usually
        else:
            print("Using CPU or CUDA not available/selected.")
            torch_dtype = torch.float32
            device = torch.device('cpu') # Explicitly CPU device object
            low_cpu_mem = False # Use more CPU RAM if needed when on CPU
            device_str = 'cpu' # Normalize device string
            print(f"Using torch_dtype: {torch_dtype} on CPU")


        # model_id = "openai/whisper-large-v3" # large-v3 instead of turbo? check model id
        model_id = "distil-whisper/distil-large-v2" # Smaller, faster alternative for testing

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem, use_safetensors=True
        )
        model.to(device) # Move model to the target device
        # Optional: Compile model for potential speedup (requires PyTorch 2.0+)
        # try:
        #     model = torch.compile(model)
        #     print("Model compiled successfully.")
        # except Exception as e:
        #      print(f"Model compilation failed: {e}")

        processor = AutoProcessor.from_pretrained(model_id)

        # Use device index for pipeline if CUDA, otherwise -1 for CPU
        pipeline_device_arg = device if 'cuda' in device_str else 'cpu'


        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=pipeline_device_arg, # Pass torch.device object
            # chunk_length_s=30, # Optional: Adjust chunk length
            # stride_length_s=5,  # Optional: Adjust stride
        )
        model_cache[device_str] = pipe
        print(f"Model initialized successfully for {device_str}.")
        return pipe
    except Exception as e:
        print(f"FATAL: Failed to initialize model on device {device_str}: {e}")
        # Remove potentially corrupted cache entry
        if device_str in model_cache:
            del model_cache[device_str]
        # Important: Reraise the exception so the calling route knows initialization failed
        raise RuntimeError(f"Model initialization failed for {device_str}") from e


# --- Transcription Core Logic ---
def process_transcription(audio_path, device_str, language, translate, transcription_id, task_id=None, original_filename="Unknown"):
    """
    Processes a single audio file for transcription.
    Assumes audio_path points to a valid audio file (e.g., WAV).
    Handles model loading, transcription, saving results, and task status updates.
    """
    processing_started = False
    try:
        print(f"[Task {task_id or 'Sync'}] Processing {original_filename} on {device_str}...")

        # Get the model pipeline for the specified device
        # This will initialize if not already cached
        pipe = initialize_model(device_str)
        processing_started = True # Mark that we at least started processing

        # Prepare generation arguments
        generate_kwargs = {}
        if language and language != 'auto':
            generate_kwargs['language'] = language
            print(f"[Task {task_id or 'Sync'}] Language set to: {language}")
        if translate:
            generate_kwargs['task'] = 'translate'
            print(f"[Task {task_id or 'Sync'}] Translation to English enabled.")

        # Perform transcription
        print(f"[Task {task_id or 'Sync'}] Starting transcription for {audio_path}...")
        # result = pipe(audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)
        # Simpler call for potentially faster inference if timestamps aren't strictly needed now
        result = pipe(audio_path, generate_kwargs=generate_kwargs)
        transcription_text = result["text"]
        print(f"[Task {task_id or 'Sync'}] Transcription completed.")

        # Save transcription to file
        transcription_path = os.path.join(TRANSCRIPTION_FOLDER, f'{transcription_id}.txt')
        os.makedirs(os.path.dirname(transcription_path), exist_ok=True) # Ensure dir exists
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text)
        print(f"[Task {task_id or 'Sync'}] Transcription saved to {transcription_path}")

        # Update task status if asynchronous
        if task_id:
            with tasks_lock:
                if task_id in tasks: # Check if task still exists
                    tasks[task_id]['status'] = 'completed'
                    tasks[task_id]['transcription'] = transcription_text
                    tasks[task_id]['id'] = transcription_id
                    # tasks[task_id]['filename'] = original_filename # Filename already set
                    print(f"[Task {task_id}] Status updated to completed.")
                else:
                    print(f"[Task {task_id}] Warning: Task ID not found during completion update.")

        return transcription_text # Return text for synchronous case

    except Exception as e:
        print(f"[Task {task_id or 'Sync'}] Error during transcription processing: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

        if task_id:
            with tasks_lock:
                 if task_id in tasks: # Check if task still exists
                    tasks[task_id]['status'] = 'error'
                    tasks[task_id]['error'] = str(e)
                    # tasks[task_id]['filename'] = original_filename # Filename already set
                    print(f"[Task {task_id}] Status updated to error.")
                 else:
                    print(f"[Task {task_id}] Warning: Task ID not found during error update.")
        else:
            # --- CRITICAL FIX for synchronous calls ---
            # Re-raise the exception so the calling route knows about the failure
            raise e

    finally:
        # --- Cleanup ---
        # Delete the intermediate audio file passed to this function
        # (Could be the original upload or the extracted WAV)
        if audio_path and os.path.exists(audio_path):
             try:
                 print(f"[Task {task_id or 'Sync'}] Cleaning up intermediate file: {audio_path}")
                 os.remove(audio_path)
             except OSError as err:
                 print(f"[Task {task_id or 'Sync'}] Warning: Could not delete intermediate file {audio_path}: {err}")
        # Note: The original uploaded file (if different from audio_path, e.g., video)
        # should be cleaned up by the calling function (transcribe or transcribe_finalize_helper)


def convert_to_wav(input_path, output_path):
    """Converts an input file (audio/video) to WAV format using ffmpeg."""
    try:
        print(f"Converting {input_path} to WAV at {output_path}...")
        # -y: Overwrite output without asking
        # -i: Input file
        # -acodec pcm_s16le: Standard WAV audio codec (signed 16-bit little-endian PCM)
        # -ar 16000: Resample audio to 16kHz (required by Whisper)
        # -ac 1: Convert to mono audio (Whisper works best with mono)
        command = [
            FFMPEG_PATH, '-y', '-i', input_path,
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, encoding='utf-8')
        print("FFmpeg conversion successful.")
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
        # Clean up potentially incomplete output file
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"FFmpeg conversion failed: {e.stderr[:500]}") # Include part of stderr
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")
         # Clean up potentially incomplete output file
        if os.path.exists(output_path):
            os.remove(output_path)
        raise Exception(f"Audio conversion failed: {e}")


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    # Update available devices each time the page is loaded
    global available_devices, default_device
    available_devices = get_available_devices()
    if len(available_devices) > 1:
        default_device = available_devices[1][0] # Default to first GPU if available
    else:
        default_device = 'cpu'
    return render_template_string(HTML, available_devices=available_devices, default_device=default_device)

# --- Non-Chunked Transcription ---

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Handles single file uploads (synchronous or asynchronous)."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Use form data for parameters in non-chunked uploads
    device_str = request.form.get('device', default_device)
    language = request.form.get('language', 'auto')
    translate = request.form.get('translate', 'false').lower() == 'true'
    polling = request.form.get('polling', 'false').lower() == 'true'
    original_filename = file.filename

    # Save the uploaded file securely
    upload_id = str(uuid.uuid4())
    _, file_extension = os.path.splitext(original_filename)
    # Include original filename (sanitized) for easier identification? Maybe just use UUID.
    saved_filename = f"{upload_id}{file_extension}"
    raw_upload_path = os.path.join(UPLOAD_FOLDER, saved_filename)

    audio_to_process = None # Path to the final audio file (WAV) for transcription
    try:
        file.save(raw_upload_path)
        print(f"File saved to {raw_upload_path}")

        # Check if conversion is needed (video or non-wav audio)
        is_video = file_extension.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv']
        is_common_audio = file_extension.lower() in ['.mp3', '.m4a', '.ogg', '.flac'] # Add other audio types if needed
        is_wav = file_extension.lower() == '.wav'

        if is_video or is_common_audio or not is_wav:
            wav_filename = f"{upload_id}.wav"
            audio_to_process = os.path.join(UPLOAD_FOLDER, wav_filename)
            if not convert_to_wav(raw_upload_path, audio_to_process):
                 # convert_to_wav now raises exceptions on failure
                 raise Exception("Audio conversion failed.") # Should be caught below
            print(f"File converted to WAV: {audio_to_process}")
        else:
            # It's already a WAV file (or assumed to be processable)
            audio_to_process = raw_upload_path
            print(f"File is WAV or assumed processable: {audio_to_process}")


        transcription_id = str(uuid.uuid4()) # Unique ID for the transcription result

        if polling:
            # --- Asynchronous Processing ---
            task_id = str(uuid.uuid4())
            with tasks_lock:
                tasks[task_id] = {
                    'status': 'processing',
                    'transcription': None,
                    'id': None, # Will be set on completion
                    'error': None,
                    'filename': original_filename, # Store original filename for status updates
                    'start_time': threading.Timer(0, lambda: None) # Placeholder for potential timing
                }
            print(f"[Task {task_id}] Created for async processing of {original_filename}")

            # Start background thread
            # Pass audio_to_process, transcription_id, task_id, original_filename
            thread = threading.Thread(target=process_transcription,
                                      args=(audio_to_process, device_str, language, translate,
                                            transcription_id, task_id, original_filename))
            thread.daemon = True # Allow app to exit even if threads are running
            thread.start()

            # Important: Don't delete raw_upload_path yet if it's different from audio_to_process
            # process_transcription will delete audio_to_process. We need to delete raw_upload_path eventually.
            # Let's handle raw_upload_path cleanup after the thread is *known* to have started
            # or potentially add it to the task dict for later cleanup?
            # For now, let's assume process_transcription handles the file it receives.
            # If conversion happened, raw_upload_path (video/other audio) still needs cleanup.
            if raw_upload_path != audio_to_process and os.path.exists(raw_upload_path):
                 print(f"[Task {task_id}] Scheduling cleanup for original upload: {raw_upload_path}")
                 # Maybe use a timer or a separate cleanup mechanism? For simplicity now:
                 try:
                    os.remove(raw_upload_path)
                 except OSError as e:
                    print(f"Warning: Failed to clean up raw upload {raw_upload_path}: {e}")


            return jsonify({"task_id": task_id}), 202 # Accepted

        else:
            # --- Synchronous Processing ---
            print(f"Starting synchronous processing for {original_filename}...")
            transcription_text = process_transcription(
                audio_path=audio_to_process, # Pass the path to the WAV/processable audio
                device_str=device_str,
                language=language,
                translate=translate,
                transcription_id=transcription_id,
                task_id=None, # Indicate synchronous call
                original_filename=original_filename
            )
            # Cleanup the original raw upload if conversion happened
            if raw_upload_path != audio_to_process and os.path.exists(raw_upload_path):
                try:
                    os.remove(raw_upload_path)
                except OSError as e:
                    print(f"Warning: Failed to clean up raw upload {raw_upload_path}: {e}")

            print(f"Synchronous processing complete for {original_filename}")
            # process_transcription already saved the file.
            # Return the transcription text and ID.
            return jsonify({"transcription": transcription_text, "id": transcription_id})

    except Exception as e:
        print(f"Error in /transcribe route for {original_filename}: {e}")
        import traceback
        traceback.print_exc()
         # Cleanup any created files if an error occurred before processing started
        if audio_to_process and os.path.exists(audio_to_process):
            try: os.remove(audio_to_process)
            except OSError: pass
        if raw_upload_path and os.path.exists(raw_upload_path):
            try: os.remove(raw_upload_path)
            except OSError: pass
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


# This route is redundant if /transcribe handles the 'polling' parameter
# @app.route('/transcribe_async', methods=['POST'])
# def transcribe_async():
#     # Kept for potential API compatibility, but logic is in /transcribe
#     # Ensure 'polling=true' is implicitly set or handled
#     # For simplicity, let's just forward or duplicate logic if needed.
#     # Or better: have the JS always call /transcribe and set the polling form data.
#     # The current JS seems to call /transcribe_async? Let's adjust JS instead.
#     # return transcribe() # This would re-read form data etc.
#     # Let's remove this route and have JS call /transcribe?
#     # Okay, keeping it for now as JS calls it, but it just forwards.
#     # This requires the form data to be sent correctly by JS.
#     # We need to ensure JS sends all params to /transcribe_async as well.
#     # It seems the JS *does* send the parameters.
#     # Let's explicitly set polling=True here.
#     request.form = request.form.copy() # Make form mutable if needed
#     request.form['polling'] = 'true' # Force polling=true for this route
#     return transcribe()


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
    # Use zero-padded index for correct sorting later
    chunk_filename = os.path.join(temp_dir, f'chunk_{chunk_index:06d}')
    try:
        chunk.save(chunk_filename)
        print(f"Received chunk {chunk_index + 1}/{total_chunks} for fileId {file_id}")

        # --- Store parameters with the last chunk ---
        # We need device, language, translate for the finalization step.
        # Let's store them in a JSON file within the temp chunk directory
        # when the last chunk arrives.
        if chunk_index == total_chunks - 1:
            params = {
                'device': request.form.get('device', default_device),
                'language': request.form.get('language', 'auto'),
                'translate': request.form.get('translate', 'false').lower() == 'true',
                'polling': request.form.get('polling', 'false').lower() == 'true',
                'original_filename': request.form.get('originalFilename', f"{file_id}_chunked_file") # Get filename from JS
            }
            params_path = os.path.join(temp_dir, 'params.json')
            with open(params_path, 'w') as f:
                json.dump(params, f)
            print(f"Parameters saved for fileId {file_id}")

        return jsonify({"message": f"Chunk {chunk_index + 1}/{total_chunks} received"}), 200
    except Exception as e:
        print(f"Error saving chunk {chunk_index} for fileId {file_id}: {e}")
        return jsonify({"error": "Failed to save chunk"}), 500

# This route might be redundant now if transcribe_chunk handles param saving
# @app.route('/transcribe_chunk_async', methods=['POST'])
# def transcribe_chunk_async():
#     # Kept for potential compatibility, assumes JS calls this for async chunks
#     # The logic is the same as transcribe_chunk, including saving params on last chunk
#     return transcribe_chunk()


def reconstruct_file_from_chunks(file_id, temp_dir):
    """Reconstructs the original file from saved chunks."""
    reconstructed_path = None # Initialize path
    try:
        chunk_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('chunk_')])
        if not chunk_files:
            raise ValueError("No chunk files found.")

        # Determine the original filename/extension if possible from params
        params_path = os.path.join(temp_dir, 'params.json')
        original_filename = f"{file_id}_reconstructed" # Default
        if os.path.exists(params_path):
             with open(params_path, 'r') as f:
                params = json.load(f)
                original_filename = params.get('original_filename', original_filename)

        _, file_extension = os.path.splitext(original_filename)
        if not file_extension: file_extension = ".bin" # Default extension if unknown

        # Use a unique name for the reconstructed file in the uploads folder
        reconstructed_filename = f"{file_id}_reconstructed{file_extension}"
        reconstructed_path = os.path.join(UPLOAD_FOLDER, reconstructed_filename)

        print(f"Reconstructing file {file_id} to {reconstructed_path}...")
        with open(reconstructed_path, 'wb') as outfile:
            for chunk_file in chunk_files:
                chunk_path = os.path.join(temp_dir, chunk_file)
                with open(chunk_path, 'rb') as infile:
                    outfile.write(infile.read())
        print(f"File reconstruction complete: {reconstructed_path}")
        return reconstructed_path, original_filename

    except Exception as e:
        print(f"Error reconstructing file {file_id}: {e}")
        # Clean up partially reconstructed file if it exists
        if reconstructed_path and os.path.exists(reconstructed_path):
            try: os.remove(reconstructed_path)
            except OSError: pass
        raise # Re-raise the exception


# Consolidate finalize logic
def transcribe_finalize_logic(file_id):
    """Handles reconstruction and transcription initiation for chunked uploads."""
    temp_dir = os.path.join(TEMP_CHUNK_FOLDER, file_id)
    reconstructed_file_path = None
    audio_to_process = None
    original_filename = f"{file_id}_chunked_file" # Default

    if not os.path.isdir(temp_dir):
        print(f"Error: Chunk directory not found for fileId {file_id}")
        raise FileNotFoundError("Chunks not found. Upload may be incomplete or expired.")

    try:
        # Load parameters saved with the last chunk
        params_path = os.path.join(temp_dir, 'params.json')
        if not os.path.exists(params_path):
             raise FileNotFoundError("Parameters file (params.json) not found in chunk directory.")
        with open(params_path, 'r') as f:
            params = json.load(f)

        device_str = params.get('device', default_device)
        language = params.get('language', 'auto')
        translate = params.get('translate', False)
        polling = params.get('polling', False)
        original_filename = params.get('original_filename', original_filename)

        # Reconstruct the file
        reconstructed_file_path, _ = reconstruct_file_from_chunks(file_id, temp_dir)
        # _, original_filename was also returned, but we already got it from params

        # --- Conversion Logic (similar to non-chunked) ---
        _, file_extension = os.path.splitext(reconstructed_file_path)
        is_video = file_extension.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv']
        is_common_audio = file_extension.lower() in ['.mp3', '.m4a', '.ogg', '.flac']
        is_wav = file_extension.lower() == '.wav'

        if is_video or is_common_audio or not is_wav:
            wav_filename = f"{file_id}_reconstructed.wav"
            audio_to_process = os.path.join(UPLOAD_FOLDER, wav_filename)
            if not convert_to_wav(reconstructed_file_path, audio_to_process):
                 raise Exception("Audio conversion failed during finalization.")
            print(f"Reconstructed file converted to WAV: {audio_to_process}")
        else:
            audio_to_process = reconstructed_file_path
            print(f"Reconstructed file is WAV or assumed processable: {audio_to_process}")
        # --- End Conversion Logic ---


        transcription_id = str(uuid.uuid4())

        if polling:
            # --- Asynchronous Finalization ---
            task_id = str(uuid.uuid4())
            with tasks_lock:
                tasks[task_id] = {
                    'status': 'processing',
                    'transcription': None,
                    'id': None,
                    'error': None,
                    'filename': original_filename,
                    'start_time': threading.Timer(0, lambda: None) # Placeholder
                }
            print(f"[Task {task_id}] Created for async finalization of chunked file {original_filename} (ID: {file_id})")

            thread = threading.Thread(target=process_transcription,
                                      args=(audio_to_process, device_str, language, translate,
                                            transcription_id, task_id, original_filename))
            thread.daemon = True
            thread.start()

             # Cleanup reconstructed file if conversion happened
            if reconstructed_file_path != audio_to_process and os.path.exists(reconstructed_file_path):
                print(f"[Task {task_id}] Scheduling cleanup for reconstructed file: {reconstructed_file_path}")
                try: os.remove(reconstructed_file_path)
                except OSError as e: print(f"Warning: Failed to clean up {reconstructed_file_path}: {e}")

            # Cleanup temp chunk dir after thread starts
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up chunk directory: {temp_dir}")
            except OSError as e:
                print(f"Warning: Failed to clean up chunk directory {temp_dir}: {e}")


            return {"task_id": task_id}, 202 # Accepted

        else:
            # --- Synchronous Finalization ---
            print(f"Starting synchronous finalization for chunked file {original_filename} (ID: {file_id})...")
            transcription_text = process_transcription(
                audio_path=audio_to_process,
                device_str=device_str,
                language=language,
                translate=translate,
                transcription_id=transcription_id,
                task_id=None,
                original_filename=original_filename
            )

            # Cleanup reconstructed file if conversion happened
            if reconstructed_file_path != audio_to_process and os.path.exists(reconstructed_file_path):
                try: os.remove(reconstructed_file_path)
                except OSError as e: print(f"Warning: Failed to clean up {reconstructed_file_path}: {e}")

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
        import traceback
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
        return jsonify({"error": f"Finalization failed: {str(e)}"}), 500

@app.route('/transcribe_finalize_async', methods=['POST'])
def transcribe_finalize_async():
    """Asynchronous finalization endpoint."""
    data = request.get_json()
    if not data or 'fileId' not in data:
        return jsonify({"error": "fileId missing in request body"}), 400
    file_id = data['fileId']
    try:
        # Force polling=True if called via this route? The logic relies on params.json now.
        # The JS should ensure polling=true is set correctly when sending chunks if this route is used.
        result, status_code = transcribe_finalize_logic(file_id)
        # Ensure the result is a task_id if successful async
        if status_code == 202 and 'task_id' not in result:
             print("Error: Async finalize logic didn't return task_id correctly.")
             return jsonify({"error": "Internal server error: async task creation failed"}), 500
        return jsonify(result), status_code
    except FileNotFoundError as e:
         return jsonify({"error": str(e)}), 404 # Chunks or params not found
    except Exception as e:
        return jsonify({"error": f"Async finalization failed: {str(e)}"}), 500


# --- Status and Download ---

@app.route('/status/<task_id>')
def status(task_id):
    """Checks the status of an asynchronous task."""
    with tasks_lock:
        task = tasks.get(task_id)
        if not task:
            return jsonify({"status": "not_found", "error": "Task ID not found or expired."}), 404

        # Return a copy to avoid modifying the original dict outside the lock
        task_snapshot = task.copy()

    # Process the snapshot outside the lock
    response = {"status": task_snapshot['status'], "filename": task_snapshot.get('filename', 'Unknown')}
    if task_snapshot['status'] == 'completed':
        response["transcription"] = task_snapshot['transcription']
        response["id"] = task_snapshot['id']
        # Optionally remove completed task after a delay?
    elif task_snapshot['status'] == 'error':
        response["error"] = task_snapshot['error']
        # Optionally remove failed task after a delay?

    return jsonify(response)


@app.route('/download/<transcription_id>')
def download(transcription_id):
    """Downloads the transcription text file."""
    # Basic security check: ensure transcription_id looks like a UUID and filename is simple
    try:
        uuid.UUID(transcription_id) # Validate UUID format
        filename = f"{transcription_id}.txt"
        if ".." in filename or "/" in filename or "\\" in filename: # Prevent directory traversal
             raise ValueError("Invalid transcription ID format.")
    except ValueError:
        return jsonify({"error": "Invalid request"}), 400

    transcription_path = os.path.join(TRANSCRIPTION_FOLDER, filename)

    if not os.path.exists(transcription_path):
        return jsonify({"error": "Transcription file not found."}), 404

    try:
        return send_file(transcription_path, as_attachment=True, download_name=f"{transcription_id}_transcription.txt")
    except Exception as e:
        print(f"Error sending file {transcription_path}: {e}")
        return jsonify({"error": "Could not send file."}), 500


# --- HTML Template (Minor Adjustments Needed in JS) ---
# The HTML itself looks mostly fine, but the JavaScript needs updates
# to handle the parameter passing for chunked uploads correctly.

HTML = '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Turbo 文字起こし</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        /* Add spinner styles */
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border-left-color: #09f;
            animation: spin 1s ease infinite;
            display: inline-block; /* Make it inline */
            margin-right: 8px; /* Add some space */
            vertical-align: middle; /* Align with text */
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        /* Style for disabled button */
        #submitButton:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
    </style>
</head>
<body class="bg-gray-100 p-8">
    <div class="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-md">
        <h1 class="text-3xl font-bold mb-6">Whisper 文字起こし</h1>
        <form id="uploadForm" class="mb-8">
            <!-- Device Selection -->
            <div class="mb-4">
                <label for="deviceSelect" class="block text-sm font-medium text-gray-700 mb-2">デバイスを選択</label>
                <select id="deviceSelect" name="device" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
                    {% for device_id, device_name in available_devices %}
                    <option value="{{ device_id }}" {% if device_id == default_device %}selected{% endif %}>{{ device_name }}</option>
                    {% endfor %}
                </select>
            </div>
            <!-- Polling Option -->
            <div class="mb-4">
                <label class="inline-flex items-center">
                    <input type="checkbox" id="pollingCheck" name="polling" value="true" class="form-checkbox h-5 w-5 text-indigo-600">
                    <span class="ml-2 text-gray-700">非同期モード (ポーリング)</span>
                </label>
            </div>
            <!-- File Input -->
            <div class="mb-4">
                <label for="fileInput" class="block text-sm font-medium text-gray-700 mb-2">音声/動画ファイルを選択 (複数可)</label>
                <input type="file" id="fileInput" name="file" accept="audio/*,video/*" multiple required class="block w-full text-sm text-gray-500
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-full file:border-0
                    file:text-sm file:font-semibold
                    file:bg-indigo-50 file:text-indigo-700
                    hover:file:bg-indigo-100
                ">
            </div>
            <!-- Language Selection -->
            <div class="mb-4">
                <label for="languageSelect" class="block text-sm font-medium text-gray-700 mb-2">言語を選択 (任意)</label>
                <select id="languageSelect" name="language" class="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
                    <option value="auto">自動検出</option>
                    <option value="ja">日本語</option>
                    <option value="en">英語</option>
                    <option value="zh">中国語</option>
                    <option value="ko">韓国語</option>
                    <option value="fr">フランス語</option>
                    <option value="de">ドイツ語</option>
                    <option value="es">スペイン語</option>
                    <!-- Add more languages as needed -->
                </select>
            </div>
            <!-- Translate Option -->
            <div class="mb-4">
                <label class="inline-flex items-center">
                    <input type="checkbox" id="translateCheck" name="translate" value="true" class="form-checkbox h-5 w-5 text-indigo-600">
                    <span class="ml-2 text-gray-700">英語に翻訳 (言語が英語以外の場合)</span>
                </label>
            </div>
            <!-- Chunk Upload Settings -->
            <div class="mb-4">
                <label class="inline-flex items-center">
                    <input type="checkbox" id="chunkUploadCheck" class="form-checkbox h-5 w-5 text-indigo-600" checked>
                    <span class="ml-2 text-gray-700">チャンクアップロードを有効にする (大きなファイル向け)</span>
                </label>
            </div>
            <div class="mb-4" id="chunkSizeContainer">
                <label for="chunkSizeInput" class="block text-sm font-medium text-gray-700 mb-2">チャンクサイズ (MB)</label>
                <input type="number" id="chunkSizeInput" min="1" max="100" value="50" class="mt-1 block w-full pl-3 pr-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm">
                <p class="mt-1 text-xs text-gray-500">サーバーのメモリに応じて調整してください (推奨: 10-100MB)。</p>
            </div>
            <!-- Submit Button -->
            <button type="submit" id="submitButton" class="w-full flex justify-center items-center bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline">
                <span id="submitButtonText">文字起こし開始</span>
                <div id="submitSpinner" class="spinner" style="display: none;"></div>
            </button>
        </form>
        <!-- Results Area -->
        <div id="results" class="space-y-4"></div>
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
        const formElements = uploadForm.elements; // Get all form elements

        chunkUploadCheck.addEventListener('change', (e) => {
            chunkSizeContainer.style.display = e.target.checked ? 'block' : 'none';
        });

        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            if (fileInput.files.length === 0) {
                alert('ファイルを選択してください');
                return;
            }

            setFormDisabled(true);
            resultsDiv.innerHTML = ''; // Clear previous results

            const useChunkUpload = chunkUploadCheck.checked;
            const chunkSizeMB = parseInt(chunkSizeInput.value, 10) || 50;
            const chunkSize = chunkSizeMB * 1024 * 1024; // Bytes

            const usePolling = pollingCheck.checked;
            const device = deviceSelect.value;
            const language = languageSelect.value;
            const translate = translateCheck.checked;

            // Process each file
            for (let file of fileInput.files) {
                const resultContainerId = `result-${generateUUID()}`;
                const resultDiv = createResultContainer(file.name, resultContainerId);
                resultsDiv.appendChild(resultDiv);
                updateResultStatus(resultContainerId, file.name, "アップロード準備中...");

                try {
                    if (usePolling) {
                        // --- Asynchronous (Polling) Path ---
                        let taskId;
                        if (useChunkUpload) {
                            updateResultStatus(resultContainerId, file.name, "チャンクアップロード中...");
                            // Pass all params needed for finalization
                            const finalResponse = await uploadFileInChunks(file, chunkSize, device, language, translate, true, file.name, resultContainerId); // Pass polling=true
                            taskId = finalResponse.task_id;
                            updateResultStatus(resultContainerId, file.name, `ファイル結合・処理待機中 (Task ID: ${taskId})...`);
                        } else {
                            updateResultStatus(resultContainerId, file.name, "アップロード中...");
                            // Use /transcribe with polling=true flag
                            const response = await uploadFileStandard(file, device, language, translate, true, resultContainerId); // Pass polling=true
                            taskId = response.task_id;
                             updateResultStatus(resultContainerId, file.name, `処理待機中 (Task ID: ${taskId})...`);
                        }
                        pollTranscription(taskId, file.name, resultContainerId); // Start polling
                    } else {
                        // --- Synchronous Path ---
                         let transcriptionData;
                        if (useChunkUpload) {
                            updateResultStatus(resultContainerId, file.name, "チャンクアップロード中...");
                             // Pass all params needed for finalization
                            transcriptionData = await uploadFileInChunks(file, chunkSize, device, language, translate, false, file.name, resultContainerId); // Pass polling=false
                            updateResultStatus(resultContainerId, file.name, "ファイル結合・文字起こし中..."); // Update status before final result
                            // The result is directly returned
                        } else {
                            updateResultStatus(resultContainerId, file.name, "アップロード＆文字起こし中...");
                            transcriptionData = await uploadFileStandard(file, device, language, translate, false, resultContainerId); // Pass polling=false
                        }
                         displayResult(resultContainerId, file.name, transcriptionData);
                    }
                } catch (error) {
                    console.error("Error processing file:", file.name, error);
                    displayError(resultContainerId, file.name, error.message || '不明なエラーが発生しました');
                }
            } // End loop through files

            setFormDisabled(false); // Re-enable form after all files are processed
        });

        function setFormDisabled(disabled) {
            for (let element of formElements) {
                element.disabled = disabled;
            }
            submitSpinner.style.display = disabled ? 'inline-block' : 'none';
            submitButtonText.textContent = disabled ? '処理中' : '文字起こし開始';
        }

        function createResultContainer(filename, containerId) {
            const div = document.createElement('div');
            div.id = containerId;
            div.className = 'bg-gray-50 p-4 rounded-lg shadow';
            div.innerHTML = `
                <h3 class="font-bold mb-2 text-gray-800">${filename}</h3>
                <div class="status-message text-gray-600">
                    <div class="spinner" style="display: none;"></div>
                    <span>初期化中...</span>
                </div>
                <div class="progress-bar-container mt-2" style="display: none;">
                    <div class="w-full bg-gray-200 rounded-full h-2.5">
                        <div class="progress-bar bg-indigo-600 h-2.5 rounded-full" style="width: 0%"></div>
                    </div>
                    <span class="progress-text text-xs text-gray-500">0%</span>
                </div>
                <div class="result-content mt-2" style="display: none;">
                    <p class="mb-2 whitespace-pre-wrap"></p> <!-- pre-wrap to preserve whitespace -->
                    <div class="action-buttons">
                        <button onclick="copyToClipboard(this)" class="bg-green-500 hover:bg-green-700 text-white font-bold py-1 px-2 text-sm rounded mr-2">
                            コピー
                        </button>
                        <a href="#" class="download-link bg-blue-500 hover:bg-blue-700 text-white font-bold py-1 px-2 text-sm rounded" style="display:none;">
                            ダウンロード (.txt)
                        </a>
                    </div>
                </div>
                <div class="error-message text-red-500 mt-2" style="display: none;"></div>
            `;
            return div;
        }

        function updateResultStatus(containerId, filename, message, showSpinner = true) {
            const container = document.getElementById(containerId);
            if (!container) return;
            const statusDiv = container.querySelector('.status-message');
            const spinner = statusDiv.querySelector('.spinner');
            const span = statusDiv.querySelector('span');
            spinner.style.display = showSpinner ? 'inline-block' : 'none';
            span.textContent = message;
            statusDiv.style.display = 'block'; // Ensure status is visible
            container.querySelector('.result-content').style.display = 'none';
            container.querySelector('.error-message').style.display = 'none';
            container.querySelector('.progress-bar-container').style.display = 'none'; // Hide progress initially
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
        }


        function displayResult(containerId, filename, data) {
            const container = document.getElementById(containerId);
            if (!container) return;
            container.querySelector('.status-message').style.display = 'none';
            container.querySelector('.error-message').style.display = 'none';
            container.querySelector('.progress-bar-container').style.display = 'none';

            const resultContent = container.querySelector('.result-content');
            resultContent.querySelector('p').textContent = data.transcription || '（文字起こし結果が空です）';
            const downloadLink = resultContent.querySelector('.download-link');
            if (data.id) {
                 downloadLink.href = `/download/${data.id}`;
                 downloadLink.style.display = 'inline-block'; // Show download link
            } else {
                 downloadLink.style.display = 'none';
            }
            resultContent.style.display = 'block';
        }

        function displayError(containerId, filename, errorMsg) {
             const container = document.getElementById(containerId);
            if (!container) return;
            container.querySelector('.status-message').style.display = 'none';
            container.querySelector('.result-content').style.display = 'none';
            container.querySelector('.progress-bar-container').style.display = 'none';

            const errorDiv = container.querySelector('.error-message');
            errorDiv.textContent = `エラー: ${errorMsg}`;
            errorDiv.style.display = 'block';
        }


        // --- Upload Functions ---

        async function uploadFileStandard(file, device, language, translate, polling, containerId) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('device', device);
            formData.append('language', language);
            formData.append('translate', translate);
            formData.append('polling', polling); // Tell backend if polling is expected

            // Use XMLHttpRequest for progress tracking
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open('POST', '/transcribe', true); // Always POST to /transcribe

                xhr.upload.onprogress = (event) => {
                    if (event.lengthComputable) {
                        const percentComplete = (event.loaded / event.total) * 100;
                        updateUploadProgress(containerId, percentComplete);
                        if (percentComplete < 100) {
                             updateResultStatus(containerId, file.name, `アップロード中... ${Math.round(percentComplete)}%`, true);
                        } else {
                             updateResultStatus(containerId, file.name, `アップロード完了、処理中...`, true);
                        }
                    }
                };

                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            resolve(JSON.parse(xhr.responseText));
                        } catch (e) {
                             reject(new Error("サーバーからの応答が無効です: " + xhr.responseText));
                        }
                    } else {
                        let errorMsg = `サーバーエラー (${xhr.status})`;
                        try {
                            const errorData = JSON.parse(xhr.responseText);
                            errorMsg = errorData.error || errorMsg;
                        } catch (e) { /* Ignore parse error, use status text */ }
                        reject(new Error(errorMsg));
                    }
                };

                xhr.onerror = () => {
                    reject(new Error('ネットワークエラーが発生しました'));
                };

                xhr.send(formData);
                updateResultStatus(containerId, file.name, `アップロード開始...`, true);
            });
        }


        async function uploadFileInChunks(file, chunkSize, device, language, translate, polling, originalFilename, containerId) {
            const totalChunks = Math.ceil(file.size / chunkSize);
            const fileId = generateUUID();
            let chunksUploaded = 0;

            console.log(`Uploading ${originalFilename} in ${totalChunks} chunks (ID: ${fileId})`);

            for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
                const start = chunkIndex * chunkSize;
                const end = Math.min(start + chunkSize, file.size);
                const chunk = file.slice(start, end);

                const formData = new FormData();
                formData.append('file', chunk);
                formData.append('fileId', fileId);
                formData.append('chunkIndex', chunkIndex);
                formData.append('totalChunks', totalChunks);
                 // Include other params with *every* chunk, backend will use the last one's params.json
                formData.append('device', device);
                formData.append('language', language);
                formData.append('translate', translate);
                formData.append('polling', polling); // Important for backend params.json
                formData.append('originalFilename', originalFilename); // Send original filename

                const chunkEndpoint = '/transcribe_chunk'; // Always use the same chunk endpoint

                try {
                    // Use fetch for chunk uploads as progress isn't per-chunk critical here
                    const response = await fetch(chunkEndpoint, {
                        method: 'POST',
                        body: formData
                        // No need for Content-Type header, FormData sets it
                    });

                    if (!response.ok) {
                        let errorMsg = `チャンク ${chunkIndex + 1} のアップロード失敗 (${response.status})`;
                         try {
                            const errorData = await response.json();
                            errorMsg = errorData.error || errorMsg;
                        } catch (e) { /* Ignore */ }
                        throw new Error(errorMsg);
                    }
                    // Update progress based on chunks uploaded
                    chunksUploaded++;
                    const percentComplete = (chunksUploaded / totalChunks) * 100;
                    updateUploadProgress(containerId, percentComplete);
                    updateResultStatus(containerId, file.name, `チャンクアップロード中 (${chunksUploaded}/${totalChunks})... ${Math.round(percentComplete)}%`, true);

                    // Optional: Small delay between chunks if needed
                    // await new Promise(resolve => setTimeout(resolve, 50));

                } catch (error) {
                    console.error(`Chunk upload error: ${error.message}`);
                    throw new Error(`チャンク ${chunkIndex + 1} のアップロード中にエラー: ${error.message}`);
                }
            }

            console.log(`All chunks uploaded for ${fileId}. Finalizing...`);
            updateResultStatus(containerId, file.name, `全チャンク (${totalChunks}/${totalChunks}) アップロード完了、最終処理中...`, true);

            // Finalize the upload
            const finalizeEndpoint = polling ? '/transcribe_finalize_async' : '/transcribe_finalize';
            const finalResponse = await fetch(finalizeEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                // Body only needs fileId, other params were saved by backend
                body: JSON.stringify({ fileId: fileId })
            });

            if (!finalResponse.ok) {
                let errorMsg = `最終処理失敗 (${finalResponse.status})`;
                 try {
                    const errorData = await finalResponse.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) { /* Ignore */ }
                throw new Error(errorMsg);
            }

             return await finalResponse.json(); // Returns {task_id} for async or {transcription, id} for sync
        }

        // --- Polling ---

        async function pollTranscription(taskId, filename, containerId) {
            console.log(`Polling started for task ${taskId} (${filename})`);
            updateResultStatus(containerId, filename, `処理待機中 (Task ID: ${taskId})...`, true);

            const pollInterval = 5000; // 5 seconds
            let consecutiveErrors = 0;
            const maxErrors = 5;

            const intervalId = setInterval(async () => {
                try {
                    const response = await fetch(`/status/${taskId}`);

                    if (!response.ok) {
                        // Handle specific errors like 404 (task not found)
                        if (response.status === 404) {
                             throw new Error(`タスク ${taskId} が見つかりません。サーバーで削除された可能性があります。`);
                        }
                        throw new Error(`ステータス確認中にサーバーエラー (${response.status})`);
                    }

                    const data = await response.json();
                    consecutiveErrors = 0; // Reset error count on success

                    updateResultStatus(containerId, filename, `処理中 (ステータス: ${data.status})...`, true);

                    if (data.status === 'completed') {
                        clearInterval(intervalId);
                        console.log(`Task ${taskId} completed.`);
                         displayResult(containerId, data.filename || filename, data); // Use filename from status if available
                    } else if (data.status === 'error') {
                        clearInterval(intervalId);
                        console.error(`Task ${taskId} failed: ${data.error}`);
                         displayError(containerId, data.filename || filename, data.error || '不明なエラーが発生しました');
                    } else if (data.status === 'processing') {
                        // Continue polling
                        console.log(`Task ${taskId} is still processing...`);
                        updateResultStatus(containerId, data.filename || filename, '文字起こし処理中...', true);
                    } else if (data.status === 'not_found') {
                        // Should be caught by response.ok check, but handle explicitly
                        throw new Error(`タスク ${taskId} が見つかりません。`);
                    }

                } catch (error) {
                    console.error(`Polling error for task ${taskId}: ${error.message}`);
                    consecutiveErrors++;
                    if (consecutiveErrors >= maxErrors) {
                         clearInterval(intervalId);
                         displayError(containerId, filename, `ポーリング中に連続エラーが発生しました。処理が中断された可能性があります: ${error.message}`);
                    } else {
                        // Show transient error in status?
                        updateResultStatus(containerId, filename, `ポーリングエラー (${consecutiveErrors}/${maxErrors})... 再試行中`, true);
                    }
                }
            }, pollInterval);
        }


        // --- Utilities ---

        function generateUUID() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }

        function copyToClipboard(button) {
            // Find the paragraph within the same result-content container
            const resultContainer = button.closest('.result-content');
            if (!resultContainer) return;
            const textToCopy = resultContainer.querySelector('p')?.textContent;

            if (!textToCopy) {
                 alert('コピーするテキストが見つかりません。');
                return;
            }

            navigator.clipboard.writeText(textToCopy).then(() => {
                const originalText = button.textContent;
                button.textContent = 'コピー完了!';
                button.disabled = true;
                setTimeout(() => {
                    button.textContent = originalText;
                    button.disabled = false;
                }, 2000);
            }).catch(err => {
                console.error('Clipboard copy failed: ', err);
                alert('クリップボードへのコピーに失敗しました。');
            });
        }

    </script>
</body>
</html>
'''

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Transcription folder: {os.path.abspath(TRANSCRIPTION_FOLDER)}")
    print(f"Temp chunk folder: {os.path.abspath(TEMP_CHUNK_FOLDER)}")
    print(f"Available devices: {available_devices}")
    print(f"Default device: {default_device}")
    print(f"IMPORTANT: If using GPU, ensure your PyTorch installation is compatible with your GPU's Compute Capability!")
    # Consider adding a check here if possible
    # initialize_model(default_device) # Optional: Pre-initialize default model at startup

    # Use waitress or gunicorn for production instead of Flask development server
    # For development:
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=False is safer

    # Example using waitress (install with pip install waitress):
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=5000)