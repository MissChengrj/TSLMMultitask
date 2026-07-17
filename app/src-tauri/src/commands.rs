use tauri::command;
use crate::data::FileInfo;
use crate::data::ModelInfo;

#[command]
pub async fn read_csv_file(path: String, data_type: String) -> Result<FileInfo, String> {
    crate::data::DataProcessor::new()
        .read_csv(&path, &data_type)
        .map_err(|e| e.to_string())
}

#[command]
pub async fn read_excel_file(path: String, data_type: String) -> Result<FileInfo, String> {
    crate::data::DataProcessor::new()
        .read_excel(&path, &data_type)
        .map_err(|e| e.to_string())
}

#[command]
pub async fn run_prediction(
    path: String,
    model_name: String,
    prediction_type: String,
    columns: Vec<String>,
    covariates: Vec<String>,
    forecast_length: usize,
    confidence_interval: f64,
    anomaly_threshold: f64,
    context_length: usize,
    data_start: usize,
    data_end: usize,
    data_type: String, // 接收前端传来的数据类型
) -> Result<serde_json::Value, String> {
    crate::data::DataProcessor::new()
        .run_prediction(
            &path,
            &model_name,
            &prediction_type,
            &columns,
            &covariates,
            forecast_length,
            confidence_interval,
            anomaly_threshold,
            context_length,
            data_start,
            data_end,
            &data_type // 向下传递给底层
        )
        .map_err(|e| e.to_string())
}

#[command]
pub async fn get_models() -> Result<Vec<ModelInfo>, String> {
    Ok(crate::data::DataProcessor::new().get_models())
}