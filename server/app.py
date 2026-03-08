import os
import pandas
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from server import response
from vectorstore import set_up_collections
from retrieval import read_files

load_dotenv()

LARGE_EMBEDDING_MODEL_NAME = os.getenv("LARGE_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
OLD_DATA_2025_PDF = os.getenv("OLD_DATA_2025_PDF", "")
NEW_DATA_FOLDER = os.getenv("NEW_DATA_FOLDER")
SUMMARY_ID_TO_CNR_CSV = os.getenv("SUMMARY_ID_TO_CNR_CSV")

SUMMARY_OUTPUT_2025_FOLDER = os.getenv("SUMMARY_OUTPUT_2025_FOLDER", "C:/Tejeswar/AI Research Engine/Final_Project/data_processed/summaries/2025/")
CNR_TO_PDF_PATH_CSV = os.getenv("CNR_TO_PDF_PATH_CSV")

app = Flask(__name__)
CORS(app)

mapping_df = pandas.read_csv(os.path.join(NEW_DATA_FOLDER, SUMMARY_ID_TO_CNR_CSV))
cnr_to_pdf_path_df = pandas.read_csv(os.path.join(NEW_DATA_FOLDER, CNR_TO_PDF_PATH_CSV))

embedding_model = HuggingFaceEmbeddings(model_name=LARGE_EMBEDDING_MODEL_NAME)
summary_store = set_up_collections.get_summary_store()
coarse_chunk_store = set_up_collections.get_coarse_chunk_store()


@app.route("/demo", methods = ['POST'])
def demo():
    json = request.get_json()
    print(json)
    return {"result" : "Muruga"}

@app.route("/get_demo", methods = ["GET"])
def get_demo():
    return "Server Returning"

def get_summary_id_from_cnr(cnr):
    summary_id = mapping_df[mapping_df["cnr"] == cnr]["summary_id"].item()
    return summary_id

def get_pdf_path_from_cnr(cnr):
    pdf_path = cnr_to_pdf_path_df[cnr_to_pdf_path_df["cnr"] == cnr]["path"].item()
    return pdf_path


@app.route("/query", methods = ["POST"])
def query():
    requestJson = request.get_json()
    user_query = requestJson["query"]
    print(user_query)

    # llm_response = response.get_response(user_query)
    llm_response, collective_chunk_documents = response.get_response(user_query, summary_store, coarse_chunk_store)

    print(llm_response)

    return jsonify({
        "result" : llm_response, 
        "retrieved_data" : collective_chunk_documents
    })

@app.route("/pdf/<path:filename>" , methods = ["GET"])
def serve_pdf(filename):
    pdf_path = get_pdf_path_from_cnr(filename)
    print(pdf_path, type(pdf_path))
    print(f"[SERVER] Send the PDF file -  {pdf_path}.pdf")

    base_dir = r"C:/Tejeswar/AI Research Engine/prototype 2/data/pdf/2025/english/"
    return send_from_directory(
        base_dir,
        pdf_path,
        mimetype="application/pdf"
    )

@app.route("/text/<path:filename>" , methods = ["GET"])
def serve_text(filename):
    summary_id = get_summary_id_from_cnr(filename)
    print(summary_id, type(summary_id))
    print(f"[SERVER] Send the summary file -  {summary_id}.txt")

    summary_text = read_files.read_summary_file(summary_id = summary_id)

    summary_data = {
        "summary_id" : summary_id,
        "summary_text" : summary_text
    }

    return jsonify(summary_data)
    # return send_from_directory(
    #     SUMMARY_OUTPUT_2025_FOLDER,
    #     summary_text_name,
    #     mimetype="text/plain",
    #     as_attachment=False
    # )

if __name__ == '__main__':
    app.run(debug=True, port = 5000)