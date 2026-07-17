// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod data;

fn main() {
    tauri::Builder::default()
        // 👇 新增这一行，注册系统的原生对话框插件
        .plugin(tauri_plugin_dialog::init()) 
        
        .invoke_handler(tauri::generate_handler![
            commands::read_csv_file,
            commands::read_excel_file,
            commands::get_models,
            commands::run_prediction
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}