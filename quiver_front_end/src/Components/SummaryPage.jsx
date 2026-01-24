import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";

export default function SummaryPage() {
  const { cnr } = useParams();
  const [summary, setSummary] = useState("");

  useEffect(() => {
    fetch(`http://localhost:5000/text/${cnr}`)
      .then(res => res.json())
      .then(data => setSummary(data.summary_text));
  }, [cnr]);
    
  return (
    <div className="min-h-screen bg-gray-100 p-10">
      <div className="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow">
        <h1 className="text-2xl font-bold mb-4">Case Summary</h1>
        <div className="whitespace-pre-line leading-relaxed text-gray-800">
            <ReactMarkdown>
              {summary}
            </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
