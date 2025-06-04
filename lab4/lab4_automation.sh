#!/usr/bin/env bash

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
NC='\033[0m'

INPUT_SIZES=(128 131072 134217728)
REPEATS=10
EXECUTABLES_SOURCE=(
  first=task1.cu
  second=task2.cu
  third=task3.cu
  fourth=task4.cu
)
BIN_DIR="bin_local_test"
CSV_RESULTS_FILE="made_with_love_by_demorgan.csv"

log() {
  echo -e "[${CYAN}$(date '+%F %T')${NC}] $*"
}
error() {
  echo -e "[${CYAN}$(date '+%F %T')${NC}] ${RED}ERROR:${NC} $*" >&2
  exit 1
}

log "Проверка наличия CUDA компилятора (nvcc)..."
if ! command -v nvcc &> /dev/null; then
  error "nvcc (CUDA Compiler) не найден. Убедитесь, что CUDA Toolkit установлен и nvcc в вашем PATH."
fi
log "${GREEN}nvcc найден.${NC}"

log "Создание каталога для исполняемых файлов: ${BIN_DIR}"
mkdir -p "${BIN_DIR}" || error "Не удалось создать каталог ${BIN_DIR}"

log "Инициализация CSV файла результатов: ${CSV_RESULTS_FILE}"
echo "Program,InputSize,Attempt,TimeType,ExecutionTime_s,Status,ExitCode" > "${CSV_RESULTS_FILE}"

log "Компиляция CUDA программ..."
declare -A COMPILED_EXECUTABLES

for item in "${EXECUTABLES_SOURCE[@]}"; do
  name=${item%%=*}
  src_file=${item#*=}
  output_exe="${BIN_DIR}/${name}"

  if [[ ! -f "$src_file" ]]; then
    log "${YELLOW}Исходный файл $src_file не найден, пропуск компиляции для $name.${NC}"
    continue
  fi

  log "▸ Компиляция $name из $src_file в $output_exe"
  if nvcc -arch=sm_35 -O3 "$src_file" -o "$output_exe"; then
    log "${GREEN}✅ $name скомпилирован успешно.${NC}"
    COMPILED_EXECUTABLES["$name"]="$output_exe"
  else
    log "${RED}❌ Ошибка компиляции $name из $src_file.${NC}"
    echo "$name,,N/A,CompilationFailed,,Error,N/A" >> "${CSV_RESULTS_FILE}"
  fi
done

if [ ${#COMPILED_EXECUTABLES[@]} -eq 0 ]; then
  log "${YELLOW}Ни одна программа не была успешно скомпилирована. Проверьте вывод выше и ${CSV_RESULTS_FILE}.${NC}"
fi

log "${GREEN}Компиляция завершена.${NC}"
echo "-----------------------------------------------------"

log "Запуск скомпилированных программ и сбор результатов в ${CSV_RESULTS_FILE}..."

for name_key in "${!COMPILED_EXECUTABLES[@]}"; do
  exe_path="${COMPILED_EXECUTABLES[$name_key]}"

  if [[ -z "$exe_path" || ! -x "$exe_path" ]]; then
    log "${YELLOW}Исполняемый файл для программы $name_key не найден или не является исполняемым (путь: '$exe_path'). Пропуск запуска.${NC}"
    continue
  fi

  log "🚀 ${GREEN}Запуск программы: $name_key (${exe_path})${NC}"
  for size in "${INPUT_SIZES[@]}"; do
    log "  ▸ ${YELLOW}Размер входных данных (N): $size${NC}"
    for attempt in $(seq 1 "$REPEATS"); do
      log "    ● ${CYAN}Попытка: $attempt / $REPEATS${NC}"

      program_output=""
      exit_code=0
      status_message="Success"

      program_output=$("$exe_path" -n "$size" 2>&1)
      exit_code=$?

      if [ $exit_code -ne 0 ]; then
        status_message="Error"
        log "      ${RED}✗ $name_key (N=$size, Попытка $attempt) завершился с ошибкой (код: $exit_code).${NC}"
      else
        log "      ${GREEN}✓ $name_key (N=$size, Попытка $attempt) завершился успешно.${NC}"
      fi

      found_time_in_this_run=0
      echo "$program_output" | while IFS= read -r line; do
          time_value=""
          time_type=""

          if [[ "$line" == *"Sequential time:"* ]]; then
              time_value=$(echo "$line" | grep -oP 'Sequential time: \K[0-9]+\.[0-9]+')
              time_type="Sequential"
          elif [[ "$line" == *"Parallel time:"* ]]; then
              time_value=$(echo "$line" | grep -oP 'Parallel time: \K[0-9]+\.[0-9]+')
              time_type="Parallel"
          elif [[ "$line" == *"Parallel time (CUDA):"* ]]; then
              time_value=$(echo "$line" | grep -oP 'Parallel time \(CUDA\): \K[0-9]+\.[0-9]+')
              time_type="ParallelCUDA"
          fi

          if [[ -n "$time_type" && -n "$time_value" ]]; then
              echo "$name_key,$size,$attempt,$time_type,$time_value,$status_message,$exit_code" >> "${CSV_RESULTS_FILE}"
              found_time_in_this_run=1
          elif [[ -n "$time_type" && -z "$time_value" ]]; then
               echo "$name_key,$size,$attempt,$time_type,ExtractionError,$status_message,$exit_code" >> "${CSV_RESULTS_FILE}"
               found_time_in_this_run=1
          fi
      done
    done
    echo ""
  done
  echo "-----------------------------------------------------"
done

log "${GREEN}Все тесты завершены. Результаты сохранены в ${CSV_RESULTS_FILE}${NC}"