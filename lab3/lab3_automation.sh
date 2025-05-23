#!/usr/bin/env bash
IFS=$'\n\t'   
#                      _        _
#                     | |      | |
#  _ __ ___   __ _  __| | ___  | |__  _   _
# | '_ ` _ \ / _` |/ _` |/ _ \ | '_ \| | | |
# | | | | | | (_| | (_| |  __/ | |_) | |_| |
# |_| |_| |_|\__,_|\__,_|\___| |_.__/ \__, |
#                                      __/ |
#                                     |___/
# ______
# |  _  \
# | | | |___ _ __ ___   ___  _ __ __ _  __ _ _ __
# | | | / _ \ '_ ` _ \ / _ \| '__/ _` |/ _` | '_ \
# | |/ /  __/ | | | | | (_) | | | (_| | (_| | | | |
# |___/ \___|_| |_| |_|\___/|_|  \__, |\__,_|_| |_|
#                                 __/ |
#
#                                        %                    @%@
#                                       ###                    ##%
#                                     %#*##                    #**#
#                                     #**##                    #**#%
#                                    %*=*#%                    #*+*#%
#                                   %#=:*#%                    #*+:*#
#                                   %*:.*#%                    #*+.**%
#                                  @#*.:*#%                   @#*+.**#@
#                                  %#*.=*#%                   #**+:**#%
#                                  %**-+*#%     %%######%%%% %#***+**#%
#                                  %***###%  +------------------*%####%@
#                   ##%%%          %*##%#+-------------=-----------+%#*---=#%
#                     *+------------------------------==-----------::---------+%
#                        %#*+*++++*=-----------------===------------:.:----------+
#                              #+-------=------------===-------------..:-----------=*
#                            #=---:.:---=------------==---------------..:------------+
#                          *=---:..:---+------------===----=----------:..:-------:.:--=#
#                        *=:---:..:---=#------------==----*------------:.:---==---:.:::-*
#                       #:.::-:.:----+:*------------==---*:------+------::----+----:.-.:-#
#                      #:...:-:-----+.:*------------=---+:..------+------------*------::--*
#                     #:...:-------+-..*------------+--+...-::-----------------=-------:--+@
#                    #:..:--------*:=:.=-----------=--=-+=:...=----+------------=----------*
#                    +..:-=------=..:--:=----------++*=:.......-----=-----------+-----------#
#                   %:.:-+------+.....-:=---=------+.....:.:....:---=------=----==----------#
#                   =.:--=-----=........=--=------=....:::##%@@@@@%===------=----*----------=%
#                  *::---=----=:.:+..:...+==------:..:.=%@%%%#*=*%%@%#------=----*-----------#
#                  #:----=-----..+%%#=..::*------=....:%%##=#*-.:*++%%+------+---*-----------*
#                  *:----=---+.+@@%%%%%=.:=-----=.....+-+**##***++-.#%%-------=--*=----------+
#                  +-----=--=.:%%+#*#-.==+-----=........*+++*++++=..%#:+------=--*-----------+
#                 %=-----=--+.+%=***##--#-----=.........+:-=+=+++..*:...+-----=--*-----------=
#                 %=-------++..#.+++#+++=---=:...........=****=:.........=-----*-+-----------=
#                 %=-------*=...:-+===+#---=..............................------*=-----------=
#                  +------+==.....-**--=--+................----------:.....=++=+*+=----------=
#                  *------*==.........---=................---------........::.....:=---------=
#                  #-----+=+-.........--=.........:==...........................:...*--------+
#                  *-=---+=*.:-------.--:....-=:...-.+#:.....................--:::+.+--------+
#                   =+--=+=+.:--::....-=..:*+=+*######***...................:--:..:.*--------+
#                   %=--+==*..........-=..#%%%####*++++++*:........................=---------*
#                   %---===+:.........-:.:%%%##*++++++++++*.......................-=---------#
#                  %#---=====.........:...%%%*++++++++++++*-.....................=----------=%
#                  %*---+====+............+%*+++++++++++++*-..............=...:==-----------*+
#                  *+---+=====+:...........#++++++++++++++*..............-:=*+=*------------*+
#                 %-+---+=======*..........:*+++++++++++++=.............+======*------------++
#                 +-+---*========+*..........*++++++++++=:...........:+========*-----------===#
#                %=-+---++=======++=+.........:**+=-:.............:++==========*-----------+=-*
#                #--+----+=======*+===+-.......................--+============++-----------+--+
#               %=--+----+====+==*===+==++-................-=----+============+*----------=----%
#               *---=+---*====+==*===*====+++=:......-===-------=+==+===+======*---------------*
#               *-:--+---++===*==*===*====++===+**+-------------*===+===*======*---------------+
#              @=----=-=--*===*=*+==+*=========*=--+=----------=====+===*======*--=------------+
#              #-----=-=*-=+==*=*=====++**++++*=-----++--------+*===*===*==========------------=
#              +-----=+=.+-*==+*...   ...----+===------=*-----*--+==*===*=======+==-------------#
#              =--=....+..=-+==*.       ..:---=+=+--------=-==---+*+++==*+======+++=------------+
#              =-+++*+.:...==#=+..         .----+-+---------==---+--=#+==+=======+*=------------=
#              -==-*=--=*:..:+++=..        ..=-#*====-----*%%%=--=-----=**========*+=------------=
#            @%%#*.....---+:..=**:.    .-  ..-*:=-=+=+---+%%%%%===--------=*=======+=------------=
#         @%%%%%%%%*.....:--+...++..  ..#-#%+....:--+-+-=+%%%%%*+---------...++====+=------------+
#        %%%%%%%%%%%%*.....:-=:......:*%%=.....:....:+-=+-+%%%%+=-------:.. ...=+==++=-----------+
#       %%%%%%%%%%%%%%%*:. ..-=+..  .-:.*..=##+-.   ..-..:=%%%%*------:..     ...=+=*==----------=#
#       %%%%%%%%%%%%%%%%%*....--+..    .#:.:..        .:..=%%%%#.......         ...+=+=-----------#
#  @@%%@%%%%%%%%%%%%%%%%%%%=...--+..   .*=........+-.  ..:=%%%%%=.         ..      .:#+=----------+
#@%%%%%%%%%%%%%%%%%%%%%%%%%%%-..:-+:.  .+#..###%%*:..   ..+%%%%%%.         ...   ..=%%%%%+--------=
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#...-+-. .:%.......        ..+%%%%%-.         .=...-%%%%%%%%%*-------#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+..-=:.  +-.          ......=%%%%*..        ..::%%%%%%%%%%%#-------+
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%:.-+-  ...          .*%....:%%%%..        ..#%%%%%%%%%%%%#--------+
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*.-+:.              .:%=....#%%=.      ..=%%%%%%%%%%%%%%#--------+
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--+:.  .*.         .:%=....*%*.   ...-%%%%%%%%%%%%%%%%%%#-------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*-+....#*.         ....  ..=#....:-#%%%%%%%%%%%%%%%%%%%%%=-----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%=+.. .##....=.           .:=----+%%%%%%%%%%%%%%%%%%%%%%------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%**. ..#%+:#+.           ...=--=%%%%%%%%%%%%%%%%%%%%%%-------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#=.. .:#%=..             ..+-=%%%%%%%%%%%%%%%%%%%%+--------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%=.      ...   ..==.      .=+%%%%%%%%%%%%%%%%%%%%%--------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-.    .:%+....##..      ..=%%%%%%%%%%%%%%%%%%%%%%-------
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%:.   ...:##:#+.          .=%%%%%%%%%%%%%%%%%%%%%%------
#%%%%%%%%%%%%%%%%%%%%#=%%%%%%%%%%%%%%%%%%%%%%%%..       .%=    .....    .=%%%%%%%%%%%%%%%%%%%%%%-----
#%%%%%%%%%%%%%%%%%#--=-=%%%%%%%%%%%%%%%%%%%%%%%%.       :%-  ..-%@#.     .-%%%%%%%%%%%%%%%%%%%%%#----
#%%%%%%%%%%%%%%@ *---=---#%%%%%%%%%%%%%%%%%%%%%%*..     .-%%%%%+:..       .=%%%%%%%%%%%%%%%%%%%%%%---
#%%%%%%%%%%%%%   +---==---*%%%%%%%%%%%%%%%%%%%%%%*.     .........         ..-%%%%%%%%%%%%%%%%%%%%%*--
#%%%%%%%%%%%%%@  -----*----+%%%%%%%%%%%%%%%%%%%%%%+..        ..+%%%=.       .=%%%%%%%%%%%%%%%%%%%%%#-
#%%%%%%%%%%%%%@ =-----+=----=%%%%%%%%%%%%%%%%%%%%%%+...          ......     ..+%%%%%%%%%%%%%%%%%%%%%+
#%%%%%%%%%%%%%% =----==+-----=%%%%%%%%%%%%%%%%%%%%%#=..      ...=#%%%*.      ..+%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%#-----==++----=%%%%%%%%%%%%%%%%%%%%%%==.      .*#=....*+       ..*%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%#-----===*=---=%%%%%%%%%%%%%%%%%%%%%%===..          ..#=..      ..*%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%=---====+----#%%%%%%%%%%%%%%%%%%%%%+-.-.         ..#%...       ..#%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%*-=====++---*%%%%%%%%%%%%%%%%%%%%%*:..=.      ..-%*..         ...#%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%======+=--=%%%%%%%%%%%%%%%%%%%%%#.. .=.    .=%#.       ...     :%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%+=====*---*%%%%%%%%%%%%%%%%%%%%#.. ..=..   ...        :%.     ..%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%+=====+--=%%%%%%%%%%%%%%%%%%%%%..  ..*.       .::..  .-%:     .=%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%*=====*=+%%%%%%%%%%%%%%%%%%%%%..   ..*..     ..%=.   .=#..    .=%%%%%%%%%%%%%%
# @@%%%%%%%%%%%%%%%%%%%%#=====*%%%%%%%%%%%%%%%%%%%%%%..   .:%#..     .*%:.  ..#-.     .#%%%%%%%%%%%%%
#    @%%%%%%%%%%%%%%%%%%%#====#%%%%%%%%%%%%%%%%%%%%%%..   .-%%%:..   ..*%=. ..#=.....  .#%%%%%%%%%%%%
#      @@%%%%%%%%%%%%%%%%%#==+%%%%%%%%%%%%%%%%%%%%%%%..   .:%%%%-.     .--. .+%..:+*:. .:%%%%%%%%%%%%
#         %%%%%%%%%%%%%%%%%#+%%%%%%%%%%%%%%%%%%%%%%%%..   .:%%%%%=.        .+%=...*:.....:%%%%%%%%%%%
#          %%%%%%%%%%%%%%%%%*%#+#%%%%%%%%%%%%%%%%%%%%..   ..%%%%%%*.      .#%-..:***=:.  .*%%%%%%%%%%
#          *#%%%%%%%%%%%%%%+:-..#%%%%%%%%%%%%%%%%%%%%..   ..%%%%%%%#.     .-....**#-.    ..#%%%%%%%%%
#          +-+%%%%%%%%%%%%*-..#%%%%%%%%%%%%%%%%%%%%%%..   ..%%%%%%%%#..          ...     ..:%%%%%%%%%
#          =---#%%%%%%%%%:-....#%%%%%%%%%%%%%%%%%%%%%..   ..#%%%%%%%%%:..        ...       .:%%%%%%%%
#          =----=%%%%%%-.=.....=%%%%%%%%%%%%%%%%%%%%#..   ..#%%%%%%%%%%=.      ........    ..=%%%%%%%
#          +------=+:...=.....:%%%%%%%%%%%%%%%%%%%%%#.     .*%%%%%%%%%%#+..    ..**#%*.      .*%%%%%+
#          +-------=+:::......#%%%%%%%%%%%%%%%%%%%%%#.     .+%%%%%%%%%%#.=.      ....:+**=... .*%%%==
#          +-------++:.:....-%%%%%%%%%%%%%%%%%%%%%%%#.     ..*%%%%%%%%%*..+..    .-#%=:..:*#....#*===
#          #-=-----+===....=%%%%%%%%%%%%%%%%%%%%%%%%*.      ..:#%%%%%*:....+.....%+..    ..=*.  :+===
#           ++------+==+++#%%%%%%%%%%%%%%%%%%%%%%%%%*.        ..=%#=...   .:=:......     ..:%.  .:+==
#           #=#-----+==+%%%%%%%%%%%%%%%%%%%%%%%%%%%%*.          ..:.      ..:=-..        ..+#.  ..-==
#            +%=----=++%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+.           .:.        .-*=.       ..=%-.    .-=
#            **%+----=#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+.           .:.        .-#%+..    ..=%*..    ..+
#             *##=----#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%+.           ...        ..#%%*..  ..#%=..   ...+=
#              %  =----#%%%%%%%%%%%%%%%%%%%%%%%%%%%%+.           ...        ..*%%%#.....+..     ..+==
#               #  #---%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-.           ...        ..*%%%%*..         ..*===
#                      _        _
#                     | |      | |
#  _ __ ___   __ _  __| | ___  | |__  _   _
# | '_ ` _ \ / _` |/ _` |/ _ \ | '_ \| | | |
# | | | | | | (_| | (_| |  __/ | |_) | |_| |
# |_| |_| |_|\__,_|\__,_|\___| |_.__/ \__, |
#                                      __/ |
#                                     |___/
# ______
# |  _  \
# | | | |___ _ __ ___   ___  _ __ __ _  __ _ _ __
# | | | / _ \ '_ ` _ \ / _ \| '__/ _` |/ _` | '_ \
# | |/ /  __/ | | | | | (_) | | | (_| | (_| | | | |
# |___/ \___|_| |_| |_|\___/|_|  \__, |\__,_|_| |_|
#                                 __/ |
#                                |___/
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
NC='\033[0m'
SPINNER=('⣾' '⣽' '⣻' '⢿' '⡿' '⣟' '⣯' '⣷')
VERSION="1"

log()   { echo -e "[${CYAN}$(date '+%F %T')${NC}] $*"; }
error() { log "${RED}ERROR:${NC} $*" >&2; if declare -f cleanup_resources &>/dev/null; then cleanup_resources 1; else exit 1; fi; }

INPUT_SIZES=(100 100000 100000000)
CORES=(2 4 8)
REPEATS=10
EXECUTABLES=(
  first=first.c
  second=second.c
  third=third.c
  fourth=fourth.c
)
declare -A LSF_CONFIG
LSF_CONFIG=(
  [JOB_PREFIX]="DEMORGAN_"
  [WALL_TIME]="00:05"
  [PENDING_LIMIT]=1
  [GLOBAL_LIMIT]=5
  [CHECK_INTERVAL]=5
  [MAX_CONCURRENT]=40
)

declare -A JOB_TRACKER
TOTAL_JOBS=0
TEMP_LSF_DIR="temp_lsf"
LOGS_DIR="logs"
REPORTS_DIR="reports"
BIN_DIR="bin"
MANAGER_PID=""
COMPLETED_JOBS_FILE=""

get_job_tracker_field() {
    local entry="${JOB_TRACKER[$1]}"
    if [[ -z "$entry" ]]; then echo ""; return; fi
    echo "$entry" | sed -n "s/.*$2=\([^;]*\).*/\1/p"
}

update_job_tracker_field() {
    local job_script_name="$1"
    local field_name="$2"
    local new_value="$3"
    local entry="${JOB_TRACKER[$job_script_name]}"
    local old_value
    old_value=$(get_job_tracker_field "$job_script_name" "$field_name")

    if [[ -z "$entry" ]]; then
        log "${YELLOW}Предупреждение: Попытка обновить несуществующую запись в JOB_TRACKER для $job_script_name${NC}"
        JOB_TRACKER[$job_script_name]="$field_name=$new_value;"
        return
    fi

    if [[ -z "$old_value" ]]; then
        JOB_TRACKER[$job_script_name]="${entry}${field_name}=${new_value};"
    else
        local new_entry
        new_entry=$(echo "$entry" | sed "s/$field_name=[^;]*/$field_name=$new_value/")
        if [[ "$new_entry" == "$entry" && "$old_value" != "$new_value" ]]; then
             log "${YELLOW}Предупреждение: sed не смог обновить поле '$field_name' для $job_script_name. Возможно, спецсимволы.${NC}"
             JOB_TRACKER[$job_script_name]="${entry}${field_name}=${new_value};"
        else
             JOB_TRACKER[$job_script_name]="$new_entry"
        fi
    fi
}

init_environment() {
  log "🚀 Инициализация v${VERSION}"
  command -v mpicc >/dev/null || error "mpicc (OpenMPI) не установлен."
  command -v bsub  >/dev/null || error "LSF (bsub) не доступен."
  mkdir -p "$LOGS_DIR" "$BIN_DIR" "$REPORTS_DIR" "$TEMP_LSF_DIR" || error "Не удалось создать каталоги."
  COMPLETED_JOBS_FILE="${TEMP_LSF_DIR}/completed_jobs_count.txt"
  echo "0" > "$COMPLETED_JOBS_FILE"
  log "Файл счетчика прогресса: $COMPLETED_JOBS_FILE"
}

compile_programs() {
  log "🛠 Компиляция программ через MPI..."
  local compiled_count=0
  local successfully_compiled_executables=()
  for item in "${EXECUTABLES[@]}"; do
    local exe_name=${item%%=*}
    local src_file=${item#*=}
    if [[ ! -f "$src_file" ]]; then
        log "${YELLOW}Исходный файл $src_file для $exe_name не найден, пропуск.${NC}"
        continue
    fi
    log "▸ Сборка $exe_name из $src_file"
    if mpicc -O3 "$src_file" -o "$BIN_DIR/$exe_name" -lm; then
      ((compiled_count++))
      successfully_compiled_executables+=("$item")
    else
      log "${YELLOW}Сборка $exe_name не удалась, пропуск.${NC}"
    fi
  done
  EXECUTABLES=("${successfully_compiled_executables[@]}")
  if (( compiled_count == 0 )); then
    error "Нет успешно скомпилированных программ. Завершение."
  fi
  log "✅ Скомпилировано программ: $compiled_count"
}

generate_jobs() {
  log "📦 Подготовка задач..."
  TOTAL_JOBS=0
  if [[ ${#EXECUTABLES[@]} -eq 0 || ${#INPUT_SIZES[@]} -eq 0 || ${#CORES[@]} -eq 0 || $REPEATS -lt 1 ]]; then
    log "${YELLOW}Недостаточно параметров для генерации задач.${NC}"
    return
  fi
  for item in "${EXECUTABLES[@]}"; do
    local exe_name=${item%%=*}
    for size in "${INPUT_SIZES[@]}"; do
      for core_count in "${CORES[@]}"; do
        for attempt in $(seq 1 "$REPEATS"); do
          local job_id="${LSF_CONFIG[JOB_PREFIX]}${exe_name}_${size}_${core_count}t_${attempt}"
          JOB_TRACKER["$job_id"]="lsf_id=UNKNOWN;status=PENDING_SUBMISSION"
          create_job_file "$job_id" "$exe_name" "$size" "$core_count" "$attempt"
          ((TOTAL_JOBS++))
        done
      done
    done
  done
  log "▸ Всего задач подготовлено: ${GREEN}$TOTAL_JOBS${NC}"
  if ((TOTAL_JOBS == 0)); then
    log "${YELLOW}0 задач было сгенерировано.${NC}"
  fi
}

create_job_file() {
  local job_script_name=$1
  local exe_name=$2
  local size=$3
  local core_count=$4
  local attempt=$5
  local lsf_file_path="$TEMP_LSF_DIR/$job_script_name.lsf"
  local current_dir; current_dir="$(pwd)"
  local log_file_path="$current_dir/$LOGS_DIR/$job_script_name.log"
  local err_file_path="$current_dir/$LOGS_DIR/$job_script_name.err"
  local executable_path="$current_dir/$BIN_DIR/$exe_name"

  cat > "$lsf_file_path" <<EOF
#!/usr/bin/env bash
#BSUB -J "$job_script_name"
#BSUB -n $core_count
#BSUB -W "${LSF_CONFIG[WALL_TIME]}"
#BSUB -o "$log_file_path"
#BSUB -e "$err_file_path"

module load mpi/openmpi-x86_64
mpirun -np $core_count "$executable_path" -n "$size"
EOF
  chmod +x "$lsf_file_path"
}

submit_job_to_lsf() {
    local job_script_name="$1"
    log "▸ Попытка отправки $job_script_name"
    local bsub_rc=0
    local bsub_output
    bsub_output=$(bsub < "$TEMP_LSF_DIR/$job_script_name.lsf" 2>&1) || bsub_rc=$?
    
    local lsf_actual_id=""
    lsf_actual_id=$(echo "$bsub_output" | sed -n 's/Job <\([0-9]*\)>.*/\1/p')

    if [[ $bsub_rc -eq 0 && -n "$lsf_actual_id" ]]; then
        log "🚀 $job_script_name отправлен, LSF ID: $lsf_actual_id."
        JOB_TRACKER["$job_script_name"]="lsf_id=$lsf_actual_id;status=SUBMITTED_TO_LSF"
        return 0
    else
        log "${YELLOW}⚠️ Не удалось отправить $job_script_name или извлечь LSF ID (rc=$bsub_rc, id='$lsf_actual_id'). Ошибка: $bsub_output.${NC}"
        JOB_TRACKER["$job_script_name"]="lsf_id=UNKNOWN;status=PENDING_SUBMISSION" 
        return 1
    fi
}

activate_jobs() {
  local num_to_activate=$1
  local activated_this_round=0
  for job_script_name in "${!JOB_TRACKER[@]}"; do
    if (( num_to_activate == 0 )); then break; fi
    local current_status
    current_status=$(get_job_tracker_field "$job_script_name" "status")
    if [[ "$current_status" == "PENDING_SUBMISSION" ]]; then
      if submit_job_to_lsf "$job_script_name"; then
        ((num_to_activate--))
        ((activated_this_round++))
      fi
    fi
  done
  if (( activated_this_round > 0 )); then
    log "✅ Отправлено $activated_this_round новых задач."
  fi
}

reset_job() {
  local job_script_name="$1"
  log "🔄 Переотправка $job_script_name"
  local lsf_id_to_kill
  lsf_id_to_kill=$(get_job_tracker_field "$job_script_name" "lsf_id")
  if [[ "$lsf_id_to_kill" != "UNKNOWN" && -n "$lsf_id_to_kill" ]]; then
    bkill "$lsf_id_to_kill" &>/dev/null
    sleep 1
  fi
  update_job_tracker_field "$job_script_name" "lsf_id" "UNKNOWN"
  update_job_tracker_field "$job_script_name" "status" "PENDING_SUBMISSION"
  submit_job_to_lsf "$job_script_name"
}

manage_jobs() {
  log "🚦 Менеджер задач (PID $$). Счетчик: $COMPLETED_JOBS_FILE"
  local completed_mgr_internal=0
  if [[ ! -f "$COMPLETED_JOBS_FILE" ]]; then
    error "Файл счетчика $COMPLETED_JOBS_FILE не найден!"
  fi

  while true; do
    if (( TOTAL_JOBS > 0 && completed_mgr_internal >= TOTAL_JOBS )); then
      log "Менеджер: все $TOTAL_JOBS задач отмечены как завершенные LSF. Выход."
      break
    fi
    if (( TOTAL_JOBS == 0 )); then
      log "Менеджер: нет задач. Выход."
      break
    fi

    declare -A lsf_statuses_by_id
    local lsf_ids_to_query=()
    for job_script_name in "${!JOB_TRACKER[@]}"; do
        local lsf_id
        lsf_id=$(get_job_tracker_field "$job_script_name" "lsf_id")
        local current_status
        current_status=$(get_job_tracker_field "$job_script_name" "status")
        if [[ "$lsf_id" != "UNKNOWN" && "$current_status" != "COMPLETED" && "$current_status" != "KILLED_OLD" ]]; then
            lsf_ids_to_query+=("$lsf_id")
        fi
    done

    if [[ ${#lsf_ids_to_query[@]} -gt 0 ]]; then
        local bjobs_query_string="${lsf_ids_to_query[*]}"
        local bjobs_output_for_ids
        bjobs_output_for_ids=$(bjobs -w -noheader -o "jobid stat" $bjobs_query_string 2>/dev/null || true)
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            local lsf_actual_id lsf_actual_status
            read -r lsf_actual_id lsf_actual_status <<< "$line"
            if [[ -n "$lsf_actual_id" && -n "$lsf_actual_status" ]]; then
                lsf_statuses_by_id["$lsf_actual_id"]="$lsf_actual_status"
            fi
        done <<< "$bjobs_output_for_ids"
    fi
    
    local running_in_lsf_count=0

    for job_script_name in "${!JOB_TRACKER[@]}"; do
        local current_status
        current_status=$(get_job_tracker_field "$job_script_name" "status")
        if [[ "$current_status" == "COMPLETED" || "$current_status" == "KILLED_OLD" ]]; then
          continue
        fi

        local lsf_id
        lsf_id=$(get_job_tracker_field "$job_script_name" "lsf_id")
        local job_marked_completed_lsf=0 

        if [[ "$lsf_id" == "UNKNOWN" ]]; then
            if [[ "$current_status" != "PENDING_SUBMISSION" ]]; then
                log "${YELLOW}Задача $job_script_name имеет LSF_ID=UNKNOWN, статус не PENDING ($current_status). Сброс.${NC}"
                update_job_tracker_field "$job_script_name" "status" "PENDING_SUBMISSION"
            fi
            continue 
        fi

        local lsf_actual_status="${lsf_statuses_by_id[$lsf_id]:-}"

        if [[ -n "$lsf_actual_status" ]]; then
            if [[ "$lsf_actual_status" == "DONE" || "$lsf_actual_status" == "EXIT" ]]; then
                update_job_tracker_field "$job_script_name" "status" "COMPLETED"
                job_marked_completed_lsf=1
                log "✅ $job_script_name (LSF ID $lsf_id) завершена (LSF статус: $lsf_actual_status)."
            elif [[ "$lsf_actual_status" == "PEND" ]]; then
                ((running_in_lsf_count++))
                update_job_tracker_field "$job_script_name" "status" "LSF_PEND"
            elif [[ "$lsf_actual_status" == "RUN" ]]; then
                ((running_in_lsf_count++))
                if [[ "$current_status" != "LSF_RUN" ]]; then
                    update_job_tracker_field "$job_script_name" "status" "LSF_RUN"
                    log "🏃 $job_script_name (LSF ID $lsf_id) выполняется (LSF: RUN)."
                fi
            else
                 if [[ "$current_status" != "$lsf_actual_status" ]]; then
                    log "${YELLOW}$job_script_name (LSF ID $lsf_id) статус LSF '$lsf_actual_status'. Обновляем трекер.${NC}"
                    update_job_tracker_field "$job_script_name" "status" "$lsf_actual_status"
                 fi
                 ((running_in_lsf_count++))
            fi
        else
            if [[ "$current_status" == "SUBMITTED_TO_LSF" || \
                  "$current_status" == "LSF_PEND" || \
                  "$current_status" == "LSF_RUN" ]]; then
                log "ℹ️ $job_script_name (LSF ID $lsf_id) не найдена в LSF. Предполагаем завершение."
                update_job_tracker_field "$job_script_name" "status" "COMPLETED"
                job_marked_completed_lsf=1
            fi
        fi

        if (( job_marked_completed_lsf == 1 )); then
            ((completed_mgr_internal++))
            local cf=0
            if [[ -s "$COMPLETED_JOBS_FILE" ]]; then
                read -r cf < "$COMPLETED_JOBS_FILE"
                case "$cf" in
                    ''|*[!0-9]*) cf=0 ;;
                esac
            fi
            ((cf++))
            echo "$cf" > "$COMPLETED_JOBS_FILE"
            log "Счетчик завершенных LSF задач обновлен: $cf / $TOTAL_JOBS (для: $job_script_name)"
        fi
    done

    local num_to_submit=$((LSF_CONFIG[MAX_CONCURRENT] - running_in_lsf_count))
    if (( num_to_submit > 0 )); then
        local pending_submission_count=0
        for jsn in "${!JOB_TRACKER[@]}"; do
          if [[ "$(get_job_tracker_field "$jsn" "status")" == "PENDING_SUBMISSION" ]]; then
            ((pending_submission_count++))
          fi
        done
        if (( pending_submission_count > 0 )); then
          activate_jobs "$num_to_submit"
        fi
    fi
    sleep "${LSF_CONFIG[CHECK_INTERVAL]}"
  done
}

show_progress() {
  if (( TOTAL_JOBS == 0 )); then
    log "${YELLOW}Нет задач.${NC}"
    return
  fi

  local si=0
  log "Прогресс... (${TOTAL_JOBS} задач). Счетчик: ${COMPLETED_JOBS_FILE}"
  local cff=0

  while (( cff < TOTAL_JOBS )); do
    if [[ ! -f "${COMPLETED_JOBS_FILE}" ]]; then
      log "${YELLOW}${COMPLETED_JOBS_FILE} не найден!${NC}"
      sleep "${LSF_CONFIG[CHECK_INTERVAL]}"
      continue
    fi

    read -r cff < "${COMPLETED_JOBS_FILE}"

    case "${cff}" in
      ''|*[!0-9]*)
        cff=0
        ;;
    esac

    if (( cff > TOTAL_JOBS )); then
      cff=${TOTAL_JOBS}
    fi

    local p=0
    if (( TOTAL_JOBS > 0 )); then
      p=$(( cff * 100 / TOTAL_JOBS ))
    fi

    local fl=$(( p / 2 ))
    local el=$(( 50 - fl ))
    local bf=""
    local be=""

    for i in $(seq 1 "${fl}"); do
      bf+="▰"
    done

    for i in $(seq 1 "${el}"); do
      be+="▱"
    done

    echo -ne "\r${SPINNER[${si}]} ${CYAN}Прогресс:${NC} [${GREEN}${bf}${NC}${be}] ${p}%% (${cff}/${TOTAL_JOBS})  "
    sleep 0.2

    si=$((( si + 1 ) % ${#SPINNER[@]}))
  done

  local fb=""
  for i in $(seq 1 50); do
    fb+="▰"
  done

  echo -e "\r${GREEN}✅ Прогресс:${NC} [${GREEN}${fb}${NC}] 100%% (${cff}/${TOTAL_JOBS})  "
  echo -e "${GREEN}🏁 Все ${TOTAL_JOBS} задач LSF завершены!${NC}"
}

cleanup_resources() {
  local es=${1:-0}

  log "🧹 Очистка (статус ${es})..."

  if [[ -n "${MANAGER_PID}" && -e "/proc/${MANAGER_PID}" ]]; then
    log " Стоп менеджер ${MANAGER_PID}..."
    kill "${MANAGER_PID}" &>/dev/null
    wait "${MANAGER_PID}" 2>/dev/null || log "${YELLOW}Менеджер ${MANAGER_PID} уже завершен.${NC}"
  fi

  MANAGER_PID=""

  local ids
  ids=$(bjobs -w -noheader -o "jobid" -J "${LSF_CONFIG[JOB_PREFIX]}*" 2>/dev/null || true)

  if [[ -n "${ids}" ]]; then
    log "Отмена LSF задач:"
    echo "${ids}" | while IFS= read -r id; do
      if [[ -n "${id}" ]]; then
        log "  bkill ${id}"
        bkill "${id}" &>/dev/null
      fi
    done
    sleep 1
  else
    log "Нет LSF задач для отмены."
  fi

  if [[ -n "${COMPLETED_JOBS_FILE}" && -f "${COMPLETED_JOBS_FILE}" ]]; then
    log "Удаление ${COMPLETED_JOBS_FILE}"
    rm -f "${COMPLETED_JOBS_FILE}"
  fi

  if [[ -d "${TEMP_LSF_DIR}" ]]; then
    log "Удаление ${TEMP_LSF_DIR}/"
    if [ "$(ls -A "${TEMP_LSF_DIR}")" ]; then
      rm -rf "${TEMP_LSF_DIR:?}"/*
    else
      log "${TEMP_LSF_DIR} пуст."
    fi
  fi

  log "Очистка завершена."

  if (( es != 0 && es < 128 )); then
    exit "${es}"
  fi
}

perform_full_cleanup() {
  log "⚠️ Полная очистка ${LSF_CONFIG[JOB_PREFIX]}*..."

  local ids
  ids=$(bjobs -w -noheader -o "jobid" -J "${LSF_CONFIG[JOB_PREFIX]}*" 2>/dev/null || true)

  if [[ -n "${ids}" ]]; then
    log "Убиваю LSF задачи:"
    echo "${ids}"
    bkill $(echo "${ids}" | tr '\n' ' ') &>/dev/null
    log "bkill отправлены."
    sleep 1
  else
    log "Нет LSF задач для убийства."
  fi

  COMPLETED_JOBS_FILE="${TEMP_LSF_DIR:-temp_lsf}/completed_jobs_count.txt"

  if [[ -f "${COMPLETED_JOBS_FILE}" ]]; then
    log "Удаление ${COMPLETED_JOBS_FILE}"
    rm -f "${COMPLETED_JOBS_FILE}"
  fi

  if [[ -d "${TEMP_LSF_DIR:-temp_lsf}" ]]; then
    log "Удаление ${TEMP_LSF_DIR:-temp_lsf}/"
    rm -rf "${TEMP_LSF_DIR:-temp_lsf:?}"/*
  else
    log "${TEMP_LSF_DIR:-temp_lsf} не найден."
  fi

  log "Полная очистка завершена."
}


main() {
  trap 'echo;log "${YELLOW}Прерывание...${NC}";cleanup_resources 130;exit 130' SIGINT SIGTERM
  init_environment
  compile_programs
  generate_jobs
  if((TOTAL_JOBS == 0)); then
    log "${YELLOW}Нет задач.${NC}"
    cleanup_resources 0
    exit 0
  fi
  log "🚀 Подача первых задач (${LSF_CONFIG[MAX_CONCURRENT]})"
  activate_jobs "${LSF_CONFIG[MAX_CONCURRENT]}"
  log "🚦 Запуск менеджера..."
  manage_jobs &
  MANAGER_PID=$!
  log "Менеджер запущен (PID: $MANAGER_PID)"
  show_progress
  log "Ожидание менеджера $MANAGER_PID..."
  if wait "$MANAGER_PID"; then
    log "Менеджер $MANAGER_PID завершился."
  else
    local st=$?
    log "${YELLOW}Менеджер $MANAGER_PID с кодом $st.${NC}"
  fi
  MANAGER_PID=""
  cleanup_resources 0
  log "🎉 Запуск LSF задач завершен. Запустите парсер логов для получения результатов."
}

usage(){
cat<<EOF
Использование: $(basename "$0") [--clean|--help|--test-compile]
  --clean          Очистить очередь LSF (${LSF_CONFIG[JOB_PREFIX]}*) и temp файлы.
  --help           Это сообщение.
  --test-compile   Только компиляция.
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    --clean)
      TEMP_LSF_DIR="${TEMP_LSF_DIR:-temp_lsf}"
      perform_full_cleanup
      exit 0
      ;;
    --help)
      usage
      exit 0
      ;;
    --test-compile)
      init_environment
      compile_programs
      log "Компиляция завершена."
      exit 0
      ;;
    *)
      echo -e "${RED}Опция $1 неизвестна${NC}" >&2
      usage >&2
      exit 1
      ;;
  esac
else
  main
fi