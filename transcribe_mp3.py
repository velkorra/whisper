import os
import argparse
from faster_whisper import WhisperModel
import time
import traceback
import gc  # Для очистки памяти

def transcribe_audio_faster(audio_file_path, model_size="base", language=None, device="cpu", compute_type="float32", beam_size=5, vad_filter=False, progress_callback=None):
    """
    Транскрибирует аудиофайл с помощью faster-whisper.
    """
    print(f"DEBUG: Вход в transcribe_audio_faster с файлом: {audio_file_path}")
    if not os.path.exists(audio_file_path):
        print(f"Ошибка: Файл не найден по пути: {audio_file_path}")
        return None
    
    try:
        print(f"Загрузка модели faster-whisper '{model_size}' (device: {device}, compute_type: {compute_type})...")
        
        # Добавляем параметры для оптимизации больших файлов
        model_kwargs = {
            "device": device,
            "compute_type": compute_type
        }
        
        # Для больших файлов уменьшаем использование памяти
        if device == "cuda":
            model_kwargs["device_index"] = 0  # Используем первую GPU
            
        model = WhisperModel(model_size, **model_kwargs)
        print(f"Модель '{model_size}' загружена.")
        print(f"Транскрибация файла: {audio_file_path}...")
        start_time = time.time()
        
        # Опции для транскрибации с оптимизацией для больших файлов
        # Убираем проблемные параметры
        transcribe_options = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "word_timestamps": False,  # Отключаем временные метки слов для экономии памяти
            "temperature": 0.0,  # Фиксированная температура для стабильности
            "condition_on_previous_text": False,  # Не опираться на предыдущий текст
        }
        
        # Добавляем только поддерживаемые параметры
        try:
            # Проверяем, какие параметры поддерживаются
            transcribe_options["compression_ratio_threshold"] = 2.4
            transcribe_options["no_speech_threshold"] = 0.6
            # Убираем logprob_threshold - он не поддерживается в вашей версии
        except:
            pass
        
        if language:
            transcribe_options["language"] = language
            print(f"Используется указанный язык: {language}")
            
        if vad_filter:
            print(f"VAD фильтр включен.")
            # Настройки VAD для больших файлов
            transcribe_options["vad_parameters"] = {
                "min_silence_duration_ms": 1000,
                "speech_pad_ms": 400
            }
        
        print("DEBUG: Перед вызовом model.transcribe()")
        print(f"DEBUG: Параметры транскрибации: {transcribe_options}")
        
        # model.transcribe возвращает генератор
        segments, info = model.transcribe(audio_file_path, **transcribe_options)
        
        print("DEBUG: После вызова model.transcribe(), получен info и segments.")
        print(f"Распознанный язык: '{info.language}' с вероятностью {info.language_probability:.2f}")
        print(f"Длительность аудио: {info.duration:.2f} секунд (~{info.duration/60:.1f} минут)")
        
        full_text = []
        print("\n--- Начало обработки сегментов ---")
        segment_count = 0
        last_progress_time = time.time()
        repetition_count = 0
        last_text = ""
        
        # Итерация по генератору для выполнения транскрибации
        for segment in segments:
            segment_count += 1
            current_time = time.time()
            
            # Показываем прогресс каждые 10 секунд или каждые 50 сегментов
            if (current_time - last_progress_time > 10) or (segment_count % 50 == 0):
                elapsed = current_time - start_time
                progress_percent = (segment.end / info.duration) * 100 if info.duration > 0 else 0
                print(f"ПРОГРЕСС: {segment_count} сегментов | {progress_percent:.1f}% | {elapsed:.0f}с прошло | ~{segment.end:.0f}с/{info.duration:.0f}с обработано")
                last_progress_time = current_time
                
                # Периодическая очистка памяти для больших файлов
                if segment_count % 100 == 0:
                    gc.collect()
            
            # Обнаружение зацикливания
            current_text = segment.text.strip()
            if current_text == last_text and len(current_text) > 10:
                repetition_count += 1
                if repetition_count > 3:  # Если повторение более 3 раз
                    print(f"⚠️ ОБНАРУЖЕНО ЗАЦИКЛИВАНИЕ на сегменте {segment_count}: '{current_text[:50]}...'")
                    print("Пропускаем повторяющиеся сегменты...")
                    continue  # Пропускаем этот сегмент
            else:
                repetition_count = 0
                last_text = current_text
            
            # Добавляем текст сегмента
            if current_text and current_text not in ["", " "]:  # Добавляем только непустые сегменты
                full_text.append(current_text)
            
            # Для очень больших файлов - промежуточное сохранение
            if segment_count % 500 == 0 and len(full_text) > 0:
                print(f"DEBUG: Промежуточное сохранение после {segment_count} сегментов...")
        
        print(f"DEBUG: Всего обработано сегментов: {segment_count}")
        if segment_count == 0 and info.duration > 0:
            print("DEBUG: ВНИМАНИЕ! Сегментов не найдено, хотя аудио имеет длительность!")
            print("Возможные причины:")
            print("- Аудио состоит только из тишины")
            print("- VAD фильтр слишком агрессивен")
            print("- Проблемы с форматом аудиофайла")
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"\nВремя обработки: {processing_time:.2f} секунд ({processing_time/60:.1f} минут)")
        print(f"Скорость: {info.duration/processing_time:.2f}x реального времени")
        
        result_text = "\n".join(full_text)
        print(f"Итоговая длина текста: {len(result_text)} символов")
        
        if not result_text:
            print("DEBUG: Результат пустой. Возможно, аудио без речи или все отфильтровано VAD.")
        
        # Очистка памяти
        del model
        gc.collect()
        
        print("--- Транскрибация завершена! ---")
        return result_text
        
    except KeyboardInterrupt:
        print("\n!!! ПРЕРЫВАНИЕ ПОЛЬЗОВАТЕЛЕМ (Ctrl+C) !!!")
        return None
    except MemoryError:
        print("\n!!! ОШИБКА ПАМЯТИ !!!")
        print("Попробуйте:")
        print("- Использовать меньшую модель (medium вместо large-v3)")
        print("- Использовать compute_type='int8' для экономии VRAM")
        print("- Включить VAD фильтр для удаления тишины")
        return None
    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА !!!")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {e}")
        print("--- Полный traceback: ---")
        traceback.print_exc()
        print("------------------------")
        return None

def save_text_with_backup(text, output_path):
    """Сохранение текста с созданием резервной копии"""
    try:
        # Создаем резервную копию, если файл уже существует
        if os.path.exists(output_path):
            backup_path = output_path + ".backup"
            os.rename(output_path, backup_path)
            print(f"Создана резервная копия: {backup_path}")
        
        # Сохраняем новый файл
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Текст сохранен в файл: {output_path}")
        return True
    except Exception as e:
        print(f"⚠️ Ошибка сохранения в '{output_path}': {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Транскрибирует аудиофайл в текст с помощью faster-whisper.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Путь к аудиофайлу для транскрибации."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                 "medium", "medium.en", "large-v1", "large-v2", "large-v3",
                 "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3"],
        help="Размер модели faster-whisper (по умолчанию: 'medium')."
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Код языка аудио (например, 'ru', 'en'). Если не указан, определяется автоматически."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Устройство для вычислений (по умолчанию: 'cuda')."
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        choices=["float16", "int8_float16", "int8", "float32"],
        help=("Тип вычислений (по умолчанию: 'float16' для cuda).\n"
              "Для больших файлов рекомендуется 'int8' для экономии памяти.")
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Размер пучка для декодирования (по умолчанию: 5). Для больших файлов рекомендуется 1-3 для избежания зацикливания."
    )
    parser.add_argument(
        "--vad_filter",
        action="store_true",
        help="Использовать VAD фильтр для удаления тишины (рекомендуется для больших файлов)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Путь для сохранения транскрибированного текста."
    )
    parser.add_argument(
        "--show_full_text",
        action="store_true",
        help="Показать полный текст в консоли (по умолчанию показываются только первые 200 символов)."
    )

    args = parser.parse_args()

    # Автоматические оптимизации для больших файлов
    if args.device == "cpu" and args.compute_type not in ["int8", "float32"]:
        print(f"Для CPU compute_type '{args.compute_type}' не оптимален. Устанавливаю 'int8'.")
        args.compute_type = "int8"

    # Проверяем размер файла
    try:
        file_size_mb = os.path.getsize(args.audio_file) / (1024 * 1024)
        print(f"Размер аудиофайла: {file_size_mb:.1f} МБ")
        
        if file_size_mb > 100:  # Больше 100 МБ
            print("🔍 Обнаружен большой файл. Рекомендации:")
            if not args.vad_filter:
                print("- Рекомендуется включить VAD фильтр (--vad_filter)")
            if args.compute_type == "float16" and args.device == "cuda":
                print("- Для экономии VRAM рекомендуется --compute_type int8")
            if args.beam_size > 3:
                print(f"- Для ускорения рекомендуется уменьшить --beam_size до 1-3 (сейчас: {args.beam_size})")
    except:
        pass

    print("\n" + "="*60)
    print("🎤 FASTER-WHISPER ТРАНСКРИБАТОР")
    print("="*60)
    print(f"📁 Файл: {args.audio_file}")
    print(f"🧠 Модель: {args.model_size}")
    print(f"💻 Устройство: {args.device}")
    print(f"⚙️ Тип вычислений: {args.compute_type}")
    print(f"🔍 Beam size: {args.beam_size}")
    print(f"🌐 Язык: {args.language if args.language else 'авто-определение'}")
    print(f"🔇 VAD фильтр: {'включен' if args.vad_filter else 'выключен'}")
    print("="*60)

    # Запуск транскрибации
    text_result = transcribe_audio_faster(
        args.audio_file,
        model_size=args.model_size,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter
    )

    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТ ТРАНСКРИБАЦИИ")
    print("="*60)

    if text_result is not None:
        if text_result:
            print(f"✅ Успешно! Получено {len(text_result)} символов текста")
            
            # Показываем текст
            if args.show_full_text:
                print("📝 Полный транскрибированный текст:")
                print("-" * 60)
                print(text_result)
                print("-" * 60)
            else:
                preview_length = 200
                if len(text_result) > preview_length:
                    print(f"📝 Предварительный просмотр (первые {preview_length} символов):")
                    print(f'"{text_result[:preview_length]}..."')
                    print("💡 Используйте --show_full_text для показа полного текста")
                else:
                    print("📝 Полный текст:")
                    print(f'"{text_result}"')
        else:
            print("⚠️ Транскрибация завершена, но текст пустой")
            print("Возможные причины: аудио без речи, слишком агрессивный VAD фильтр")

        # Сохранение файла
        output_txt_file_path = args.output_file
        if output_txt_file_path is None:
            base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
            output_txt_file_path = base_name + "_transcribed_fw.txt"

        if save_text_with_backup(text_result, output_txt_file_path):
            print(f"💾 Файл сохранен: {output_txt_file_path}")
        else:
            print("❌ Не удалось сохранить файл")
    else:
        print("❌ Транскрибация не удалась")
        print("Возможные решения:")
        print("- Проверьте формат аудиофайла")
        print("- Попробуйте меньшую модель")
        print("- Используйте compute_type='int8' для экономии памяти")
        print("- Включите VAD фильтр")

    print("="*60)