import React, {useState} from "react";

const openPDF = (pdfFile) => {
  const url = `http://localhost:5000/pdf/${pdfFile}`;
  window.open(url, "_blank");
};

const openText = (textFile) => {
  const url = `http://localhost:5000/text/${textFile}`;
  window.open(url, "_blank");
};

const viewSummary = (summary, cnr) => {
  if(! cnr) {
    openText(cnr);
  }  
  window.open(`/summary/${cnr}`, "_blank");
};

const SearchResults = ({ documents }) => {
  const [summary, setSummary] = useState("");

  const handleViewSummary = (summary_text, cnr) => {
  if(! summary_text){
    openPDF(cnr);
    return;
  }
  setSummary(summary_text);   // triggers rendering
  };

  return (
    <div className="h-full overflow-y-auto p-4 bg-gray-50">
      <h2 className="text-lg font-semibold mb-4">
        Retrieved Documents
      </h2>

      {documents.length === 0 && (
        <p className="text-sm text-gray-500">
          No documents retrieved yet.
        </p>
      )}

      <div className="space-y-4">
        {documents.map((doc, index) => (
          <div
            key={index}
            className="bg-white border rounded-lg p-4 shadow-sm"
          >
            <strong>{doc.respondent}</strong> <i>vs</i> <strong>{doc.petitioner}</strong>
            {/* <div className="text-sm text-gray-800 mb-2">
              <strong>Chunk Text:</strong>
              <p className="mt-1 whitespace-pre-wrap">
                {doc[0]}
              </p>
            </div> */}

            <div className="flex flex-wrap gap-2 text-xs text-gray-600 mt-2">
              <pre className=" inline-block mt-1 bg-gray-100 p-2 rounded overflow-x-auto">
                {doc.cnr}
              </pre>
              <pre className="inline-block mt-1 bg-blue-100 p-2 rounded overflow-x-auto">
                {doc.year}
              </pre>
              <pre className="inline-block mt-1 bg-pink-100 p-2 rounded overflow-x-auto">
                {doc.court}
              </pre>
              <pre className="inline-block mt-1 bg-violet-100 p-2 rounded overflow-x-auto">
                {doc.case_id}
              </pre>
              <pre className="inline-block mt-1 bg-yellow-100 p-2 rounded overflow-x-auto">
                CORAM {doc.judge}
              </pre>
              {/* <pre className="mt-1 bg-yellow-100 p-2 rounded overflow-x-auto">
                Displosal: {doc[1].disposal_nature}
              </pre> */}

              <button onClick={() => openPDF(doc.cnr)} className="mt-1 bg-green-600 text-white px-3 py-1 rounded overflow-x-auto">
                <strong>
                    View Judgement 
                </strong>
              </button>

              <button onClick={() => viewSummary(doc.summary, doc.cnr)}  className="bg-blue-600 text-white px-3 py-1 rounded" >
                <strong>
                    View Summary 
                </strong>
              </button>
              <div className="w-full" />


              <small>Supporting Chunks: {doc.supporting_chunks}</small>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SearchResults;
