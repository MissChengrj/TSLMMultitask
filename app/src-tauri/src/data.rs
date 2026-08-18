use anyhow::{Context, Result};
use calamine::{open_workbook_auto, DataType, Range, Reader};
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub filename: String,
    pub size: u64,
    pub rows: usize,
    pub columns: usize,
    pub column_names: Vec<String>,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub engine_sn: Option<String>,
    pub tail: Option<String>,
    pub eng_pos: Option<String>,
    pub op_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub description: String,
    pub supported_tasks: Vec<String>,
    pub parameters: Vec<ModelParameter>,
    pub available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameter {
    pub name: String,
    pub param_type: String,
    pub default: String,
    pub description: String,
}

pub struct DataProcessor {}

impl DataProcessor {
    pub fn new() -> Self {
        DataProcessor {}
    }

    pub fn read_csv(&self, path: &str, data_type: &str) -> Result<FileInfo> {
        let file_path = Path::new(path);
        let size = std::fs::metadata(file_path)?.len();
        let filename = file_path.file_name().unwrap().to_string_lossy().to_string();

        let mut reader = ReaderBuilder::new().has_headers(false).from_path(file_path)?;
        let records: Result<Vec<_>, _> = reader.records().collect();
        let records = records?;
        let total_rows = records.len();
        let is_qar = data_type == "qar";

        let mut header_row_idx = 0;
        let mut engine_sn = None;
        
        // 智能寻址表头与 ESN
        for r in 0..std::cmp::min(total_rows, 20) {
            if let Some(rec) = records.get(r) {
                if is_qar {
                    if let Some(val) = rec.get(1) {
                        if val.to_uppercase().contains("FRAME") || val.to_uppercase().contains("计数") { header_row_idx = r; break; }
                    }
                } else {
                    if let Some(val) = rec.get(0) {
                        let s = val.to_string();
                        let v_up = s.to_uppercase();
                        if v_up.starts_with("ESN ") {
                            engine_sn = Some(s.replace("ESN ", "").trim().to_string());
                        } else if v_up.contains("FLIGHT DATETIME") || v_up.contains("FLIGHT") {
                            header_row_idx = r; break;
                        }
                    }
                }
            }
        }
        
        // 防止 ESN 写在表头下方导致没抓取到
        if !is_qar && engine_sn.is_none() {
            for r in 0..std::cmp::min(total_rows, 10) {
                if let Some(rec) = records.get(r) {
                    if let Some(val) = rec.get(0) {
                        let s = val.to_string();
                        if s.to_uppercase().starts_with("ESN ") {
                            engine_sn = Some(s.replace("ESN ", "").trim().to_string()); break;
                        }
                    }
                }
            }
        }

        let mut start_time = None; let mut end_time = None;
        if !is_qar {
            start_time = records.get(1).and_then(|r| r.get(1)).map(|s| s.to_string());
            end_time = records.get(1).and_then(|r| r.get(2)).map(|s| s.to_string());
        }

        // ================= 剥离元数据与特征 =================
        let mut column_names = Vec::new();
        if let Some(header_rec) = records.get(header_row_idx) {
            let start_col = if is_qar { 3 } else { 4 }; // 基线数据从第5列(索引4)起提取特征
            for i in start_col..header_rec.len() {
                let name = header_rec.get(i).unwrap_or("").trim();
                if !name.is_empty() && name != "NaN" { column_names.push(name.to_string()); }
            }
        }

        let first_data_row = header_row_idx + 1;
        let last_data_row = total_rows.saturating_sub(1);
        let mut tail = None; let mut eng_pos = None; let mut op_code = None;
        
        if total_rows > first_data_row {
            if is_qar {
                start_time = records.get(first_data_row).and_then(|r| r.get(2)).map(|s| s.to_string());
                end_time = records.get(last_data_row).and_then(|r| r.get(2)).map(|s| s.to_string());
            } else {
                tail = records.get(first_data_row).and_then(|r| r.get(1)).map(|s| s.to_string());
                eng_pos = records.get(first_data_row).and_then(|r| r.get(2)).map(|s| s.to_string());
                op_code = records.get(first_data_row).and_then(|r| r.get(3)).map(|s| s.to_string());
            }
        }

        Ok(FileInfo {
            path: path.to_string(), filename, size,
            rows: total_rows.saturating_sub(first_data_row),
            columns: column_names.len(), column_names,
            start_time, end_time, engine_sn, tail, eng_pos, op_code,
        })
    }

    pub fn read_excel(&self, path: &str, data_type: &str) -> Result<FileInfo> {
        let file_path = Path::new(path);
        let size = std::fs::metadata(file_path)?.len();
        let filename = file_path.file_name().unwrap().to_string_lossy().to_string();

        let mut workbook = open_workbook_auto(file_path)?;
        let sheets = workbook.worksheets();
        let sheet = &sheets.into_iter().next().context("No sheets found")?.1;
        let (total_rows, total_cols): (usize, usize) = sheet.get_size();
        let is_qar = data_type == "qar";

        let mut header_row_idx = 0;
        let mut engine_sn = None;

        for r in 0..std::cmp::min(total_rows, 20) {
            if is_qar {
                if let Some(c) = sheet.get((r, 1)) {
                    if c.to_string().to_uppercase().contains("FRAME") || c.to_string().to_uppercase().contains("计数") { header_row_idx = r; break; }
                }
            } else {
                if let Some(c) = sheet.get((r, 0)) {
                    let s = c.to_string();
                    let v_up = s.to_uppercase();
                    if v_up.starts_with("ESN ") {
                        engine_sn = Some(s.replace("ESN ", "").trim().to_string());
                    } else if v_up.contains("FLIGHT DATETIME") || v_up.contains("FLIGHT") {
                        header_row_idx = r; break;
                    }
                }
            }
        }

        if !is_qar && engine_sn.is_none() {
            for r in 0..std::cmp::min(total_rows, 10) {
                if let Some(c) = sheet.get((r, 0)) {
                    let s = c.to_string();
                    if s.to_uppercase().starts_with("ESN ") {
                        engine_sn = Some(s.replace("ESN ", "").trim().to_string()); break;
                    }
                }
            }
        }

        let mut start_time = None; let mut end_time = None;
        if !is_qar {
            start_time = sheet.get((1, 1)).map(|c| c.to_string());
            end_time = sheet.get((1, 2)).map(|c| c.to_string());
        }

        let mut tail = None; let mut eng_pos = None; let mut op_code = None;
        let first_data_row = header_row_idx + 1;
        
        if total_rows > first_data_row {
            if is_qar {
                start_time = sheet.get((first_data_row, 2)).map(|c| c.to_string());
                end_time = sheet.get((total_rows.saturating_sub(1), 2)).map(|c| c.to_string());
            } else {
                tail = sheet.get((first_data_row, 1)).map(|c| c.to_string());
                eng_pos = sheet.get((first_data_row, 2)).map(|c| c.to_string());
                op_code = sheet.get((first_data_row, 3)).map(|c| c.to_string());
            }
        }

        let mut column_names = Vec::new();
        if total_rows > header_row_idx {
            let start_col = if is_qar { 3 } else { 4 }; // 基线数据从第5列(索引4)起提取特征
            for col in start_col..total_cols {
                if let Some(c) = sheet.get((header_row_idx, col)) {
                    let name = c.to_string();
                    if !name.trim().is_empty() && name != "NaN" { column_names.push(name.trim().to_string()); }
                }
            }
        }

        Ok(FileInfo {
            path: path.to_string(), filename, size,
            rows: total_rows.saturating_sub(first_data_row),
            columns: column_names.len(), column_names,
            start_time, end_time, engine_sn, tail, eng_pos, op_code,
        })
    }

    pub fn run_prediction(
        &self, path: &str, model_name: &str, prediction_type: &str, columns: &[String],
        covariates: &[String], forecast_length: usize, confidence_interval: f64, anomaly_threshold: f64,
        context_length: usize, data_start: usize, data_end: usize, data_type: &str,
        python_path: &str, model_dir: &str, enable_cpu_quantization: bool
    ) -> Result<serde_json::Value> {
        let cols_str = columns.join(",");
        let cov_str = if covariates.is_empty() { "-".to_string() } else { covariates.join(",") };
        let project_root = std::env::var("TSLM_PROJECT_ROOT").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
        let script_path = project_root.join("app").join("engine").join("bridge.py");
        let model_root = {
            let candidate = PathBuf::from(model_dir);
            if candidate.is_absolute() { candidate } else { project_root.join(candidate) }
        };
        let model_path = match model_name {
            "Chronos-2" => model_root.join("chronos-2"),
            "Chronos-2-Multitask" => model_root.join("multitask_chronos2").join("checkpoint-final"),
            "Chronos-T5-Base" => model_root.join("mlm_finetuned").join("mlm-lora-finetuned"),
            _ => model_root.join("default_model"),
        };

        println!("🚀 启动 Python 引擎...");
        let python_executable = if python_path.trim().is_empty() { "python" } else { python_path.trim() };
        let output = Command::new(python_executable)
            .arg(&script_path).arg(prediction_type).arg(path).arg(&cols_str).arg(&cov_str)
            .arg(&model_path).arg(forecast_length.to_string()).arg(confidence_interval.to_string())
            .arg(anomaly_threshold.to_string()).arg(context_length.to_string())
            .arg(data_start.to_string()).arg(data_end.to_string()).arg(data_type)
            .arg(enable_cpu_quantization.to_string())
            .output()?;

        let output_str = String::from_utf8_lossy(&output.stdout);
        let json_result: serde_json::Value = serde_json::from_str(&output_str)
            .map_err(|_| anyhow::anyhow!("解析失败: {}", output_str))?;

        if json_result["status"] == "success" { Ok(json_result) } else { Err(anyhow::anyhow!("{}", json_result["message"])) }
    }

    pub fn get_models(&self, model_dir: &str) -> Vec<ModelInfo> {
        let project_root = std::env::var("TSLM_PROJECT_ROOT").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
        let model_root = {
            let candidate = PathBuf::from(model_dir);
            if candidate.is_absolute() { candidate } else { project_root.join(candidate) }
        };
        let chronos2_path = model_root.join("chronos-2");
        let multitask_path = model_root.join("multitask_chronos2").join("checkpoint-final");
        let chronos_t5_path = model_root.join("mlm_finetuned").join("mlm-lora-finetuned");

        vec![
            ModelInfo {
                name: "Chronos-2-Multitask".to_string(),
                description: "Chronos-2 多任务微调模型，支持预测、历史重构、数据修复和异常检测".to_string(),
                available: multitask_path.exists(),
                supported_tasks: vec![
                    "单变量预测".to_string(),
                    "多变量预测".to_string(),
                    "协变量预测".to_string(),
                    "数据修复".to_string(),
                    "异常检测".to_string(),
                ],
                parameters: vec![
                    ModelParameter { name: "prediction_length".to_string(), param_type: "integer".to_string(), default: "24".to_string(), description: "预测步数".to_string() },
                    ModelParameter { name: "context_length".to_string(), param_type: "integer".to_string(), default: "512".to_string(), description: "上下文窗口长度".to_string() },
                    ModelParameter { name: "anomaly_threshold".to_string(), param_type: "float".to_string(), default: "3.0".to_string(), description: "异常判别基准 (Sigma倍数)".to_string() },
                ],
            },
            ModelInfo {
                name: "Chronos-2".to_string(),
                description: "Chronos-2 官方预测模型，仅用于预测任务".to_string(),
                available: chronos2_path.exists(),
                supported_tasks: vec![
                    "单变量预测".to_string(),
                    "多变量预测".to_string(),
                    "协变量预测".to_string(),
                ],
                parameters: vec![
                    ModelParameter { name: "prediction_length".to_string(), param_type: "integer".to_string(), default: "24".to_string(), description: "预测步数".to_string() },
                    ModelParameter { name: "context_length".to_string(), param_type: "integer".to_string(), default: "512".to_string(), description: "上下文窗口长度".to_string() },
                ],
            },
            ModelInfo {
                name: "Chronos-T5-Base".to_string(),
                description: "Chronos-T5 基础预测模型，仅用于预测任务".to_string(),
                available: chronos_t5_path.exists(),
                supported_tasks: vec![
                    "单变量预测".to_string(),
                    "多变量预测".to_string(),
                    "协变量预测".to_string(),
                ],
                parameters: vec![
                    ModelParameter { name: "prediction_length".to_string(), param_type: "integer".to_string(), default: "24".to_string(), description: "预测步数".to_string() },
                ],
            },
        ]
    }
}
