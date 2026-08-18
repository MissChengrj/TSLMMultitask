import sys
import os
import json
import logging
import warnings
import traceback
from pathlib import Path

project_root = os.environ.get("TSLM_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
inference_dir = os.path.join(project_root, "scripts", "inference")
src_dir = os.path.join(project_root, "src")
for import_path in (src_dir, project_root, inference_dir):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

warnings.filterwarnings("ignore")
logging.getLogger("chronos").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import transformers
    transformers.logging.set_verbosity_error()
except:
    pass

try:
    from tslm_multitask.inference import InferenceRequest, MultitaskInferenceEngine
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Python 依赖导入失败: {str(e)}"}))
    sys.exit(0)


def run_task(task_type, file_path, columns, covariates_str, model_path, param_length, confidence_interval, anomaly_threshold, context_length, data_start, data_end, data_type, enable_cpu_quantization=True):
    try:
        request = InferenceRequest(
            task_type=task_type,
            file_path=file_path,
            columns=columns,
            covariates=covariates_str,
            model_path=model_path,
            prediction_length=param_length,
            confidence_interval=confidence_interval,
            anomaly_threshold=anomaly_threshold,
            context_length=context_length,
            data_start=data_start,
            data_end=data_end,
            data_type=data_type,
        )
        result_json = MultitaskInferenceEngine(
            project_root=project_root,
            enable_cpu_quantization=enable_cpu_quantization,
        ).run(request)
        print(json.dumps(result_json, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"status": "error", "message": f"引擎内部错误:\n{traceback.format_exc()}"}, ensure_ascii=False))
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
        enable_cpu_quantization = sys.argv[13].lower() == "true" if len(sys.argv) > 13 else True
        
        run_task(task_type, file_path, columns, covariates_str, model_path, param_length, confidence_interval, anomaly_threshold, context_length, data_start, data_end, data_type, enable_cpu_quantization)
    else:
        print(json.dumps({"status": "error", "message": "参数不足。"}))
        sys.exit(0)
