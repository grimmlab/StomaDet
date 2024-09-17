import gradio as gr
from pathlib import Path
from sql_fun import create_new_tables, select_joined_stoma_summary, select_joined_stoma_details, select_stoma_summary, select_stoma_details
from process_stoma_fun import filter_stoma_summary, save_all_data_as_csv, process_zip_file, create_app_parser
import shutil

args, _ = create_app_parser()
db_file_path = Path("./data/stoma.db")
save_file_path = Path("./data/")
zip_path = Path("./data/zips/")


def save_zip_from_tmp(tmp_path: Path, zip_path: Path):
    zip_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp_path, zip_path / f"{tmp_path.stem}.zip")
    return


def process_zip(tmp_path: str, progress=gr.Progress()):
    tmp_path = Path(tmp_path)
    progress(0, desc="Start Processing zip...")
    save_zip_from_tmp(tmp_path, zip_path)
    _ = process_zip_file(zip_path / f"{tmp_path.stem}.zip", db_file_path, progress)
    filter_stoma_summary(db_file_path)
    return


def select_stoma_summary_wrapper():
    return select_stoma_summary(db_file_path)


def select_stoma_details_wrapper():
    return select_stoma_details(db_file_path)


def save_all_data_as_csv_wrapper(download_summary: bool):
    if download_summary:
        print("Downloading Summary")
        select_fun = select_joined_stoma_summary
        suffix = "summary"
    else:
        print("Downloading Details")
        select_fun = select_joined_stoma_details
        suffix = "details"
    save_link = save_all_data_as_csv(db_file_path, save_file_path, select_fun, suffix)
    return save_link


if __name__ == "__main__":
    if args.delete_db is True:
        print(db_file_path.resolve())
        create_new_tables(db_file_path)
    with gr.Blocks() as demo:
        gr.Markdown("# StomaApp")
        with gr.Tab("Upload and process Files"):
            tmp_path = gr.File(label="Select a zip file with images in png format.", file_types=[".zip"])
            submit = gr.Button(value="Predict")
            submit.click(process_zip, tmp_path, tmp_path, queue=True)
        with gr.Tab("Show Summary"):
            with gr.Column():
                show_details = gr.Button(value="Show Database Summary Entries")
                df_box = gr.Dataframe(label="")
                show_details.click(select_stoma_summary_wrapper, None, [df_box])
        with gr.Tab("Show Details"):
            with gr.Row():
                with gr.Column():
                    show_details = gr.Button(value="Show Database Detail Entries")
                    df_box = gr.Dataframe(label="")
                    show_details.click(select_stoma_details_wrapper, None, [df_box])
        with gr.Tab("Download Data"):
            with gr.Row():
                with gr.Column():
                    save_function = gr.Radio(
                        label="Please select the following table. The complete table will be saved.",
                        choices=[("Summary", True), ("Details", False)], value="Summary")
                    save_csv_btn = gr.Button("Save to file...")
                    save_output = gr.File(
                        label="Output File. First press the 'Save to file...' button. Then press on the link to download.")
                    save_csv_btn.click(save_all_data_as_csv_wrapper, save_function, [save_output])

    demo.launch(allowed_paths=["./data/"], share=False, server_name="0.0.0.0", server_port=9000)
