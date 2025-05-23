#!/usr/bin/env bash
IFS=$'\n\t'

LOGS_DIR="logs"
JOB_PREFIX="DEMORGAN_" 
DEFAULT_OUTPUT_CSV="reports/summary_parsed_$(date +%Y%m%d_%H%M%S).csv"
OUTPUT_CSV="${1:-$DEFAULT_OUTPUT_CSV}"

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
NC='\033[0m'

log() { echo -e "[${CYAN}$(date '+%F %T')${NC}] $*"; }

log "🚀 Запуск парсера логов..."
log "Каталог логов: $LOGS_DIR"
log "Префикс задач: $JOB_PREFIX"
log "Выходной CSV:  $OUTPUT_CSV"

if [[ ! -d "$LOGS_DIR" ]]; then
    log "${RED}ОШИБКА: Каталог логов '$LOGS_DIR' не найден.${NC}"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "executable,parameter,cores,attempt,parallel_time,sequential_time" > "$OUTPUT_CSV"

processed_files=0
found_files=0

find "$LOGS_DIR" -maxdepth 1 -name "${JOB_PREFIX}*.log" -type f | while IFS= read -r logfile; do
    ((found_files++))
    filename=$(basename "$logfile")
    log "📄 Обработка файла: $filename"

    job_id_part=${filename%.log}
    remaining=${job_id_part#"$JOB_PREFIX"}

    exe_name=${remaining%%_*}
    remaining=${remaining#*_}
    size=${remaining%%_*}
    remaining=${remaining#*_}
    core_count=${remaining%%t_*}
    attempt=${remaining##*t_}

    if [[ -z "$exe_name" || -z "$size" || -z "$core_count" || -z "$attempt" ]]; then
        log "${YELLOW}⚠️ Не удалось разобрать параметры из имени файла '$filename'. Пропуск.${NC}"
        continue
    fi

    par_time="N/A"
    seq_time="N/A"

    if [[ -s "$logfile" ]]; then
        par_time_raw=$(grep 'Parallel time:' "$logfile" | awk '{print $3}' | tail -n 1)
        seq_time_raw=$(grep 'Sequential time:' "$logfile" | awk '{print $3}' | tail -n 1)
        
        par_time=${par_time_raw:-N/A}
        seq_time=${seq_time_raw:-N/A}

        if [[ "$par_time" == "N/A" || "$seq_time" == "N/A" ]]; then
             log "${YELLOW}    Не удалось извлечь время из '$filename'. Проверьте формат строк.${NC}"
        fi
    else
        log "${YELLOW}⚠️ Файл лога '$filename' пустой. Запись N/A.${NC}"
    fi

    echo "$exe_name,$size,$core_count,$attempt,$par_time,$seq_time" >> "$OUTPUT_CSV"
    ((processed_files++))
done

log "🏁 Обработка завершена."
log "Найдено файлов: $found_files"
log "Обработано файлов (успешно разобрано имя): $processed_files"
log "📊 Итоговый отчет сохранен в: $OUTPUT_CSV"