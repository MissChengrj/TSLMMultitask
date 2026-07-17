import sys
import os
import json
import logging
import numpy as np
import pandas as pd
import warnings
import torch
import traceback

project_root = r"E:\Cursor Code\Chronos"
inference_dir = os.path.join(project_root, "scripts", "inference")
sys.path.append(project_root)
sys.path.append(inference_dir)

warnings.filterwarnings("ignore")
logging.getLogger("chronos").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import transformers
    transformers.logging.set_verbosity_error()
except:
    pass

try:
    from chronos import Chronos2Pipeline
    from mlm_inference import detect_anomalies, interpolate_missing
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Python 依赖导入失败: {str(e)}"}))
    sys.exit(0)

def load_smart_multivariate_data(file_path, columns, data_type):
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path, header=None, low_memory=False)
    else:
        df = pd.read_excel(file_path, header=None)

    header_idx = 0
    for i in range(min(20, len(df))):
        if data_type == "qar":
            val = str(df.iloc[i, 1]).upper() if df.shape[1] > 1 else ""
            if "FRAME" in val or "计数" in val:
                header_idx = i
                break
        else:
            val = str(df.iloc[i, 0]).upper() if df.shape[1] > 0 else ""
            if "FLIGHT" in val:
                header_idx = i
                break

    df.columns = df.iloc[header_idx].astype(str).str.strip()
    df = df.iloc[header_idx + 1:].reset_index(drop=True)

    data_list = []
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"数据文件中找不到特征列: '{col}'。请检查数据类型选择是否正确。")
        col_data = pd.to_numeric(df[col], errors='coerce').fillna(0.0).values
        data_list.append(col_data)

    return np.array(data_list, dtype=np.float32)

# 注意：这里确保 run_task 接收了 data_type
def run_task(task_type, file_path, columns, covariates_str, model_path, param_length, confidence_interval, anomaly_threshold, context_length, data_start, data_end, data_type):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到本地模型权重: {model_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = Chronos2Pipeline.from_pretrained(model_path, device_map=device, dtype=torch.float32)
        if device == "cpu":
            pipeline.model = torch.quantization.quantize_dynamic(pipeline.model, {torch.nn.Linear}, dtype=torch.qint8)
        
        target_columns = [c.strip() for c in columns.split(',') if c.strip()]
        covariate_columns = [c.strip() for c in covariates_str.split(',') if c.strip() and c.strip() != '-']
        all_cols = target_columns + covariate_columns
        
        abs_file_path = file_path if os.path.isabs(file_path) else os.path.join(project_root, file_path)
        
        # ================= 核心修复：将 data_type 传给解析函数 =================
        multivariate_data = load_smart_multivariate_data(abs_file_path, all_cols, data_type)
        # ========================================================================
        
        multivariate_data = multivariate_data[:, data_start:data_end]
        
        result_json = {"status": "success", "model_used": model_path.split(os.sep)[-1], "device_used": device, "multi_results": []}

        SAFE_MAX_CONTEXT = 2048
        ctx_len = min(context_length, multivariate_data.shape[1], SAFE_MAX_CONTEXT)
        history_data = multivariate_data[:, -ctx_len:]
        history_timestamps = [f"ROW_{data_end - ctx_len + i}" for i in range(ctx_len)]
        future_timestamps = [f"PRED_{data_end + i}" for i in range(param_length)]
        
        full_timestamps = [f"ROW_{data_start + i}" for i in range(multivariate_data.shape[1])]
        CHUNK_SIZE = 512

        if task_type == "data_repair":
            for i in range(len(target_columns)):
                series = multivariate_data[i] 
                missing_indices = np.where(np.isnan(series) | (series == 0.0))[0]
                interpolated = series.copy()
                
                if len(missing_indices) > 0:
                    for start in range(0, len(series), CHUNK_SIZE):
                        end = min(start + CHUNK_SIZE, len(series))
                        chunk = series[start:end]
                        actual_len = len(chunk)
                        chunk_missing = np.where(np.isnan(chunk) | (chunk == 0.0))[0]
                        
                        if len(chunk_missing) > 0:
                            if actual_len < CHUNK_SIZE:
                                chunk_padded = np.pad(chunk, (0, CHUNK_SIZE - actual_len), mode='edge')
                            else:
                                chunk_padded = chunk

                            inputs = [{"target": np.where(np.isnan(chunk_padded), np.nan, chunk_padded)}]
                            recon_preds = pipeline.predict(inputs=inputs, prediction_length=len(chunk_padded))
                            recon_tensor = recon_preds[0].cpu().numpy() if isinstance(recon_preds, list) else recon_preds.cpu().numpy()
                            if recon_tensor.ndim == 3: recon_tensor = np.squeeze(recon_tensor, axis=0)
                            
                            recon_tensor = recon_tensor[..., :actual_len]
                            
                            chunk_interpolated = interpolate_missing(chunk, recon_tensor, chunk_missing, use_median=True)
                            interpolated[start:end] = chunk_interpolated
                            
                result_json["multi_results"].append({
                    "target_name": target_columns[i],
                    "original_data": series.tolist(),
                    "repaired_data": interpolated.tolist(),
                    "repaired_count": len(missing_indices),
                    "timestamps": full_timestamps 
                })
            
        elif task_type == "anomaly_detection":
            for i in range(len(target_columns)):
                series = multivariate_data[i]
                total_len = len(series)
                full_anomaly_scores = np.zeros(total_len)
                
                for start in range(0, total_len, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, total_len)
                    chunk = series[start:end]
                    actual_len = len(chunk)
                    
                    if actual_len < CHUNK_SIZE:
                        chunk_padded = np.pad(chunk, (0, CHUNK_SIZE - actual_len), mode='edge')
                    else:
                        chunk_padded = chunk
                        
                    inputs = [{"target": chunk_padded}]
                    recon_preds = pipeline.predict(inputs=inputs, prediction_length=len(chunk_padded))
                    recon_tensor = recon_preds[0].cpu().numpy() if isinstance(recon_preds, list) else recon_preds.cpu().numpy()
                    if recon_tensor.ndim == 3: recon_tensor = np.squeeze(recon_tensor, axis=0)
                    
                    recon_tensor = recon_tensor[..., :actual_len]
                    
                    _, chunk_scores = detect_anomalies(chunk, recon_tensor)
                    full_anomaly_scores[start:end] = chunk_scores
                
                mean_score = np.mean(full_anomaly_scores)
                std_score = np.std(full_anomaly_scores)
                dynamic_threshold = mean_score + anomaly_threshold * std_score
                reconstructed_baseline = series - full_anomaly_scores
                
                anomaly_indices = np.where(full_anomaly_scores > dynamic_threshold)[0]
                anomalies_list = [{"index": int(idx), "value": float(series[idx]), "score": float(full_anomaly_scores[idx])} for idx in anomaly_indices]
                
                result_json["multi_results"].append({
                    "target_name": target_columns[i],
                    "original_data": series.tolist(),
                    "reconstructed_data": reconstructed_baseline.tolist(),
                    "anomaly_scores": full_anomaly_scores.tolist(),
                    "threshold": float(dynamic_threshold),
                    "anomalies": anomalies_list,
                    "total_anomalies": len(anomalies_list),
                    "timestamps": full_timestamps
                })
            
        elif task_type in ["multivariate_forecast", "univariate_forecast", "covariate_forecast"]:
            if task_type == "covariate_forecast": inputs = [history_data] 
            elif task_type == "multivariate_forecast": inputs = [history_data[:len(target_columns)]] 
            else: inputs = [history_data[0]] 
            
            forecast_preds = pipeline.predict(inputs=inputs, prediction_length=param_length)
            pred_tensor = forecast_preds[0].cpu().numpy() if isinstance(forecast_preds, list) else forecast_preds.cpu().numpy()
            if pred_tensor.ndim == 2: pred_tensor = np.expand_dims(pred_tensor, axis=0)
            
            lower_q = (1.0 - confidence_interval) / 2.0 * 100
            upper_q = (1.0 + confidence_interval) / 2.0 * 100
            
            num_targets_out = min(len(target_columns), pred_tensor.shape[0])
            for i in range(num_targets_out):
                target_pred = pred_tensor[i]
                result_json["multi_results"].append({
                    "target_name": target_columns[i],
                    "predictions": np.percentile(target_pred, 50, axis=0).tolist(),
                    "confidence_lower": np.percentile(target_pred, lower_q, axis=0).tolist(),
                    "confidence_upper": np.percentile(target_pred, upper_q, axis=0).tolist(),
                    "timestamps": future_timestamps,
                    "history_data": history_data[i].tolist(),
                    "history_timestamps": history_timestamps
                })

        print(json.dumps(result_json))

    except Exception as e:
        print(json.dumps({"status": "error", "message": f"引擎内部错误:\n{traceback.format_exc()}"}))
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) > 12:
        task_type = sys.argv[1]
        file_path = sys.argv[2]
        columns = sys.argv[3]
        covariates_str = sys.argv[4]
        model_path = sys.argv[5]
        param_length = int(sys.argv[6])
        confidence_interval = float(sys.argv[7])
        anomaly_threshold = float(sys.argv[8])
        context_length = int(sys.argv[9])
        data_start = int(sys.argv[10])
        data_end = int(sys.argv[11])
        data_type = sys.argv[12]
        
        run_task(task_type, file_path, columns, covariates_str, model_path, param_length, confidence_interval, anomaly_threshold, context_length, data_start, data_end, data_type)
    else:
        print(json.dumps({"status": "error", "message": "参数不足。"}))
        sys.exit(0)