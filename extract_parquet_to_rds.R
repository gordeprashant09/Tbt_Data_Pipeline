#!/usr/bin/env Rscript

# Unified parquet → RDS extractor
#
# Script location : /media/svipl/Data/historical_data/extract_parquet_to_rds.R
#
# Usage:
#   Rscript /media/svipl/Data/historical_data/extract_parquet_to_rds.R --date 20260327 --segment CM
#   Rscript /media/svipl/Data/historical_data/extract_parquet_to_rds.R --date 20260327 --segment FUTSTK
#   Rscript /media/svipl/Data/historical_data/extract_parquet_to_rds.R --date 20260327 --segment ALL
#
# Input  (FUTSTK) : /media/svipl/Data/shared/<date>/FUTSTK/
# Input  (CM)     : /media/svipl/Data/shared/<date>/CM_Data/
#
# Output (FUTSTK) : /media/svipl/Data/shared/<date>/
# Output (CM)     : /media/svipl/Data/shared/<date>/
#
# CLI flags override env vars; env vars override built-in defaults.

SHARED_ROOT <- "/media/svipl/Data/shared"

# ── helpers ───────────────────────────────────────────────────────────────────

env_value <- function(name, default = "") {
  value <- Sys.getenv(name, unset = default)
  if (!nzchar(value)) default else value
}

env_flag <- function(name, default = FALSE) {
  value <- tolower(env_value(name, if (default) "1" else "0"))
  value %in% c("1", "true", "t", "yes", "y", "on")
}

split_env <- function(value) {
  if (!nzchar(value)) return(character())
  trimws(unlist(strsplit(value, ",", fixed = TRUE), use.names = FALSE))
}

parse_cli <- function() {
  args   <- commandArgs(trailingOnly = TRUE)
  result <- list(date = NULL, segment = NULL)
  i <- 1
  while (i <= length(args)) {
    if (args[i] %in% c("--date", "-d") && i < length(args)) {
      result$date    <- args[i + 1]; i <- i + 2
    } else if (args[i] %in% c("--segment", "--seg", "-s") && i < length(args)) {
      result$segment <- toupper(args[i + 1]); i <- i + 2
    } else {
      warning("Unknown argument: ", args[i], call. = FALSE); i <- i + 1
    }
  }
  result
}

# ── parquet / data.table helpers (shared) ────────────────────────────────────

core_columns <- c(
  "capture_ns", "exchange_ns", "seq_no", "msg_type", "side",
  "order_id", "buy_order_id", "sell_order_id", "price", "quantity",
  "stream_id", "token"
)
main_msg_types <- c("N", "M", "X", "T")
aux_msg_types  <- c("G", "H", "J", "K")

ensure_user_library <- function() {
  user_lib <- env_value("R_LIBS_USER", "")
  if (!nzchar(user_lib)) {
    minor    <- sub("\\..*$", "", R.version$minor)
    user_lib <- file.path("~", "R",
                          paste0(R.version$platform, "-library"),
                          paste0(R.version$major, ".", minor))
  }
  user_lib <- path.expand(user_lib)
  dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(unique(c(user_lib, .libPaths())))
  invisible(user_lib)
}

install_missing_packages <- function(packages) {
  user_lib <- ensure_user_library()
  missing  <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (!length(missing)) return(invisible(TRUE))
  if (!env_flag("INSTALL_PACKAGES", FALSE)) {
    stop("Missing R package(s): ", paste(missing, collapse = ", "),
         "\nRe-run with INSTALL_PACKAGES=1 to install.", call. = FALSE)
  }
  repos <- env_value("CRAN_REPOS", "https://cloud.r-project.org")
  install.packages(missing, repos = repos, lib = user_lib)
  still <- missing[!vapply(missing, requireNamespace, logical(1), quietly = TRUE)]
  if (length(still)) stop("Install failed: ", paste(still, collapse = ", "), call. = FALSE)
}

ensure_parquet_reader <- function() {
  if (requireNamespace("arrow",        quietly = TRUE)) return("arrow")
  if (!requireNamespace("nanoparquet", quietly = TRUE)) install_missing_packages("nanoparquet")
  if (requireNamespace("nanoparquet",  quietly = TRUE)) return("nanoparquet")
  stop("Install R package 'arrow' or 'nanoparquet'.", call. = FALSE)
}

read_parquet_data <- function(path) {
  if (requireNamespace("arrow",       quietly = TRUE)) return(arrow::read_parquet(path, as_data_frame = TRUE))
  if (requireNamespace("nanoparquet", quietly = TRUE)) return(nanoparquet::read_parquet(path))
  stop("No parquet reader available.", call. = FALSE)
}

clean_names    <- function(x) gsub("[^a-z0-9]+", "_", tolower(x))

rename_aliases <- function(df) {
  names(df) <- clean_names(names(df))
  aliases <- list(
    capture_ns    = c("capture_ns","capturens","capture_time_ns","capture_timestamp_ns","packet_capture_ns"),
    exchange_ns   = c("exchange_ns","exchangens","exchange_time_ns","exchange_timestamp_ns","exch_ns"),
    seq_no        = c("seq_no","seqno","sequence_no","sequence_number","packet_seq_no"),
    msg_type      = c("msg_type","msgtype","message_type","type","message"),
    side          = c("side","order_side"),
    order_id      = c("order_id","orderid"),
    buy_order_id  = c("buy_order_id","buyorderid","buy_id","buyid"),
    sell_order_id = c("sell_order_id","sellorderid","sell_id","sellid"),
    price         = c("price","order_price"),
    quantity      = c("quantity","qty","order_quantity"),
    stream_id     = c("stream_id","streamid","stream","feed_id","feedid","channel_id","channelid"),
    token         = c("token","instrument_token","instrumenttoken","security_token","securitytoken")
  )
  for (target in names(aliases)) {
    if (target %in% names(df)) next
    hit <- aliases[[target]][aliases[[target]] %in% names(df)]
    if (length(hit)) names(df)[match(hit[[1]], names(df))] <- target
  }
  df
}

as_clean_id <- function(x) {
  if (inherits(x, "integer64"))  out <- as.character(x)
  else if (is.factor(x))         out <- as.character(x)
  else if (is.numeric(x))        out <- ifelse(is.na(x), "0", sprintf("%.0f", x))
  else                           out <- trimws(as.character(x))
  out[is.na(out)] <- "0"
  out[tolower(out) %in% c("","na","nan","null")] <- "0"
  out <- sub("\\.0+$", "", out)
  out[out == ""] <- "0"
  out
}

is_blankish <- function(x) {
  if (inherits(x, "integer64") || is.numeric(x)) return(is.na(x))
  s <- tolower(trimws(as.character(x)))
  is.na(x) | s %in% c("","na","nan","null")
}

prep_tbt <- function(df) {
  df       <- rename_aliases(as.data.frame(df, stringsAsFactors = FALSE))
  required <- setdiff(core_columns, c("stream_id","token"))
  missing  <- setdiff(required, names(df))
  if (length(missing)) stop("missing column(s): ", paste(missing, collapse=", "), call.=FALSE)
  if (!"stream_id" %in% names(df)) df$stream_id <- NA
  if (!"token"     %in% names(df)) df$token     <- NA
  df <- df[core_columns]
  df$order_id_was_na      <- is_blankish(df$order_id)
  df$buy_order_id_was_na  <- is_blankish(df$buy_order_id)
  df$sell_order_id_was_na <- is_blankish(df$sell_order_id)
  df$msg_type      <- as.character(df$msg_type)
  df$side          <- as.character(df$side)
  df$order_id      <- as_clean_id(df$order_id)
  df$buy_order_id  <- as_clean_id(df$buy_order_id)
  df$sell_order_id <- as_clean_id(df$sell_order_id)
  df
}

sort_tbt <- function(df) {
  if (requireNamespace("data.table", quietly = TRUE)) {
    data.table::setDT(df)
    data.table::setorderv(df, c("capture_ns","seq_no"), na.last = TRUE)
    df <- as.data.frame(df)
  } else {
    df <- df[order(df$capture_ns, df$seq_no, na.last = TRUE), , drop = FALSE]
    rownames(df) <- NULL
  }
  df
}

parse_symbol_filter <- function() {
  symbols <- split_env(env_value("SYMBOL_FILTER", ""))
  if (!length(symbols)) return(character())
  if (length(symbols) == 1 && symbols %in% c("ALL","*")) return(character())
  unique(symbols)
}

# ── FUTSTK segment ────────────────────────────────────────────────────────────

expiry_time <- function(e, tz="Asia/Kolkata") as.POSIXct(as.numeric(e), origin="1980-01-01", tz=tz)
expiry_date <- function(e, tz="Asia/Kolkata") as.Date(expiry_time(e, tz=tz))

parse_parquet_manifest_futstk <- function(input_dir, trade_date) {
  files   <- list.files(input_dir, pattern="\\.parquet$", recursive=TRUE, full.names=TRUE)
  if (!length(files)) stop("No parquet files found under: ", input_dir, call.=FALSE)
  pattern <- sprintf("^%s_FUTSTK_(.*)_([0-9]+)\\.parquet$", trade_date)
  parsed  <- regexec(pattern, basename(files))
  pieces  <- regmatches(basename(files), parsed)
  ok      <- lengths(pieces) == 3
  if (!any(ok)) stop("No filenames matched: ", trade_date, "_FUTSTK_<SYMBOL>_<EXPIRY>.parquet", call.=FALSE)
  data.frame(
    path         = files[ok],
    symbol       = vapply(pieces[ok], `[[`, character(1), 2),
    expiry_epoch = vapply(pieces[ok], `[[`, character(1), 3),
    stringsAsFactors = FALSE
  )
}

select_expiry_epoch <- function(manifest, trade_date, requested_expiry="") {
  available <- sort(unique(manifest$expiry_epoch))
  if (nzchar(requested_expiry)) {
    if (!requested_expiry %in% available)
      stop("EXPIRY_EPOCH=", requested_expiry, " not found. Available: ",
           paste(available, collapse=", "), call.=FALSE)
    return(requested_expiry)
  }
  trade_day <- as.Date(trade_date, format="%Y%m%d")
  valid     <- available[expiry_date(available) >= trade_day]
  if (!length(valid)) stop("No expiry on/after ", trade_date, call.=FALSE)
  valid[which.min(expiry_date(valid))]
}

export_one_futstk <- function(path, symbol, output_dir, trade_date, overwrite) {
  out_path <- file.path(output_dir, sprintf("TBT_%s_%s.rds", symbol, trade_date))
  aux_path <- file.path(output_dir, "aux", sprintf("TBT_%s_%s_aux.rds", symbol, trade_date))
  if (file.exists(out_path) && file.exists(aux_path) && !overwrite) {
    message("  [skip] exists: ", out_path)
    return(data.frame(symbol=symbol, rows=NA_integer_, status="exists", path=out_path))
  }
  message("  reading : ", path)
  prepped <- prep_tbt(read_parquet_data(path))
  out     <- sort_tbt(prepped[prepped$msg_type %in% main_msg_types,,drop=FALSE])
  aux     <- sort_tbt(prepped[prepped$msg_type %in% aux_msg_types, ,drop=FALSE])
  dir.create(output_dir,          recursive=TRUE, showWarnings=FALSE)
  dir.create(dirname(aux_path),   recursive=TRUE, showWarnings=FALSE)
  saveRDS(out, out_path, compress="xz")
  saveRDS(aux, aux_path, compress="xz")
  message("  wrote   : ", out_path, "  rows=", nrow(out))
  data.frame(symbol=symbol, rows=nrow(out), status="written", path=out_path)
}

run_futstk <- function(trade_date) {
  # Input  : /media/svipl/Data/shared/<date>/FUTSTK/
  # Output : /media/svipl/Data/shared/<date>/
  input_dir        <- env_value("FUTSTK_INPUT_DIR",
                        file.path(SHARED_ROOT, trade_date, "FUTSTK"))
  requested_expiry <- env_value("EXPIRY_EPOCH", "")
  overwrite        <- env_flag("OVERWRITE", FALSE)
  symbols_req      <- parse_symbol_filter()

  manifest     <- parse_parquet_manifest_futstk(input_dir, trade_date)
  expiry_epoch <- select_expiry_epoch(manifest, trade_date, requested_expiry)

  output_dir   <- env_value("FUTSTK_OUTPUT_DIR",
                    file.path(SHARED_ROOT, trade_date))

  selected <- manifest[manifest$expiry_epoch == expiry_epoch,,drop=FALSE]
  selected <- selected[!duplicated(selected$symbol),,drop=FALSE]
  if (length(symbols_req)) selected <- selected[selected$symbol %in% symbols_req,,drop=FALSE]
  if (!nrow(selected)) stop("No FUTSTK files matched symbols/expiry.", call.=FALSE)

  message("")
  message("┌─ FUTSTK ─────────────────────────────────────────────")
  message("│  trade_date   : ", trade_date)
  message("│  expiry_epoch : ", expiry_epoch,
          "  (", format(expiry_date(expiry_epoch), "%d-%b-%Y"), ")")
  message("│  input_dir    : ", input_dir)
  message("│  output_dir   : ", output_dir)
  message("│  symbols      : ", nrow(selected))
  message("└───────────────────────────────────────────────────────")

  results <- vector("list", nrow(selected))
  for (i in seq_len(nrow(selected))) {
    results[[i]] <- tryCatch(
      export_one_futstk(selected$path[[i]], selected$symbol[[i]],
                        output_dir, trade_date, overwrite),
      error = function(e) {
        warning("FAILED ", selected$symbol[[i]], ": ", conditionMessage(e), call.=FALSE)
        data.frame(symbol=selected$symbol[[i]], rows=NA_integer_,
                   status="error", path=selected$path[[i]])
      }
    )
  }
  summary      <- do.call(rbind, results)
  summary_path <- file.path(output_dir,
                    sprintf("extract_futstk_summary_%s.csv", trade_date))
  dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
  write.csv(summary, summary_path, row.names=FALSE)
  message("Summary CSV : ", summary_path)
  print(summary)
  invisible(summary)
}

# ── CM (Cash) segment ─────────────────────────────────────────────────────────

parse_parquet_manifest_cash <- function(input_dir, trade_date, series_filter="EQ") {
  files   <- list.files(input_dir, pattern="\\.parquet$", recursive=FALSE, full.names=TRUE)
  if (!length(files)) stop("No parquet files found under: ", input_dir, call.=FALSE)
  pattern <- sprintf("^%s_(.*)_([^_]+)\\.parquet$", trade_date)
  parsed  <- regexec(pattern, basename(files))
  pieces  <- regmatches(basename(files), parsed)
  ok      <- lengths(pieces) == 3
  if (!any(ok)) stop("No filenames matched: ", trade_date, "_<SYMBOL>_<SERIES>.parquet", call.=FALSE)
  manifest <- data.frame(
    path   = files[ok],
    symbol = vapply(pieces[ok], `[[`, character(1), 2),
    series = vapply(pieces[ok], `[[`, character(1), 3),
    stringsAsFactors = FALSE
  )
  if (nzchar(series_filter) && !series_filter %in% c("ALL","*")) {
    wanted   <- trimws(strsplit(series_filter, ",", fixed=TRUE)[[1]])
    manifest <- manifest[manifest$series %in% wanted,,drop=FALSE]
  }
  if (!nrow(manifest)) stop("No cash files matched SERIES_FILTER=", series_filter, call.=FALSE)
  manifest[order(manifest$symbol, manifest$series, manifest$path),,drop=FALSE]
}

export_one_cash <- function(path, symbol, series, output_dir, trade_date, overwrite) {
  out_path <- file.path(output_dir, sprintf("TBT_%s_%s.rds", symbol, trade_date))
  aux_path <- file.path(output_dir, "aux", sprintf("TBT_%s_%s_aux.rds", symbol, trade_date))
  if (file.exists(out_path) && file.exists(aux_path) && !overwrite) {
    message("  [skip] exists: ", out_path)
    return(data.frame(symbol=symbol, series=series, rows=NA_integer_, status="exists", path=out_path))
  }
  message("  reading : ", path)
  prepped <- prep_tbt(read_parquet_data(path))
  out     <- sort_tbt(prepped[prepped$msg_type %in% main_msg_types,,drop=FALSE])
  aux     <- sort_tbt(prepped[prepped$msg_type %in% aux_msg_types, ,drop=FALSE])
  dir.create(output_dir,        recursive=TRUE, showWarnings=FALSE)
  dir.create(dirname(aux_path), recursive=TRUE, showWarnings=FALSE)
  saveRDS(out, out_path, compress="xz")
  saveRDS(aux, aux_path, compress="xz")
  message("  wrote   : ", out_path, "  rows=", nrow(out))
  data.frame(symbol=symbol, series=series, rows=nrow(out), status="written", path=out_path)
}

run_cash <- function(trade_date) {
  # Input  : /media/svipl/Data/shared/<date>/CM_Data/
  # Output : /media/svipl/Data/shared/<date>/
  input_dir     <- env_value("CM_INPUT_DIR",
                     file.path(SHARED_ROOT, trade_date, "CM_Data"))
  output_dir    <- env_value("CM_OUTPUT_DIR",
                     file.path(SHARED_ROOT, trade_date))
  series_filter <- env_value("SERIES_FILTER", "EQ")
  overwrite     <- env_flag("OVERWRITE", FALSE)
  symbols_req   <- parse_symbol_filter()

  manifest <- parse_parquet_manifest_cash(input_dir, trade_date, series_filter)
  if (length(symbols_req)) manifest <- manifest[manifest$symbol %in% symbols_req,,drop=FALSE]
  if (!nrow(manifest)) stop("No cash files matched symbols/series.", call.=FALSE)

  message("")
  message("┌─ CM (Cash) ───────────────────────────────────────────")
  message("│  trade_date    : ", trade_date)
  message("│  series_filter : ", series_filter)
  message("│  input_dir     : ", input_dir)
  message("│  output_dir    : ", output_dir)
  message("│  symbols       : ", nrow(manifest))
  message("└───────────────────────────────────────────────────────")

  results <- vector("list", nrow(manifest))
  for (i in seq_len(nrow(manifest))) {
    results[[i]] <- tryCatch(
      export_one_cash(manifest$path[[i]], manifest$symbol[[i]], manifest$series[[i]],
                      output_dir, trade_date, overwrite),
      error = function(e) {
        warning("FAILED ", manifest$symbol[[i]], ": ", conditionMessage(e), call.=FALSE)
        data.frame(symbol=manifest$symbol[[i]], series=manifest$series[[i]],
                   rows=NA_integer_, status="error", path=manifest$path[[i]])
      }
    )
  }
  summary      <- do.call(rbind, results)
  summary_path <- file.path(output_dir, sprintf("extract_cm_summary_%s.csv", trade_date))
  dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
  write.csv(summary, summary_path, row.names=FALSE)
  message("Summary CSV : ", summary_path)
  print(summary)
  invisible(summary)
}

# ── entry point ───────────────────────────────────────────────────────────────

main <- function() {
  cli <- parse_cli()

  trade_date <- if (!is.null(cli$date))    cli$date    else env_value("TRADE_DATE", "")
  segment    <- if (!is.null(cli$segment)) cli$segment else toupper(env_value("SEGMENT", "CM"))

  if (!nzchar(trade_date))
    stop("--date is required. Example: --date 20260327", call.=FALSE)
  if (!grepl("^[0-9]{8}$", trade_date))
    stop("Invalid --date '", trade_date, "'. Expected YYYYMMDD format.", call.=FALSE)

  valid_segs <- c("CM","FUTSTK","ALL")
  if (!segment %in% valid_segs)
    stop("Invalid --segment '", segment, "'. Choose from: ", paste(valid_segs, collapse=", "), call.=FALSE)

  parquet_reader <- ensure_parquet_reader()
  if (env_flag("USE_DATATABLE", TRUE)) {
    install_missing_packages("data.table")
    data.table::setDTthreads(as.integer(env_value("DATA_TABLE_THREADS", "1")))
  }

  message("=== extract_parquet_to_rds ===")
  message("trade_date     : ", trade_date)
  message("segment        : ", segment)
  message("parquet_reader : ", parquet_reader)
  message("shared_root    : ", SHARED_ROOT)

  if (segment %in% c("FUTSTK","ALL")) run_futstk(trade_date)
  if (segment %in% c("CM","ALL"))     run_cash(trade_date)

  message("")
  message("=== Done ===")
}

if (sys.nframe() == 0) main()
