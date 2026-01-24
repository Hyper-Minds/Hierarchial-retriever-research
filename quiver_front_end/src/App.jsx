import React, { useState } from "react";
import SearchBar from "./Components/Searchbar";
import ChatContent from "./Components/ChatContent";
import SearchResults from "./Components/Searchresults";
import SummaryPage from "./Components/SummaryPage";
import { Routes, Route } from "react-router-dom";

const App = () => {
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);

  const handleSearch = async (query) => {
    setMessages((prev) => [...prev, { role: "user", text: query }]);

    try {
      const response = await fetch("http://localhost:5000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();
      const llm_response = data.result;
      const retrieved_doc_data_array = data.retrieved_data;

      setMessages((prev) => [...prev, { role: "bot", text: llm_response }]);
      setDocuments(retrieved_doc_data_array);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Error connecting to server." },
      ]);
    }
  };

  return (
    <Routes>
      {/* Home page */}
      <Route
        path="/"
        element={
          <div className="h-screen flex flex-col bg-gray-100">
            <header className="p-4 bg-black text-white text-xl font-bold">
              Quiver
            </header>

            <div className="flex flex-1 overflow-hidden">
              <div className="w-[60%] border-r">
                <ChatContent messages={messages} />
              </div>

              <div className="w-[40%]">
                <SearchResults documents={documents} />
              </div>
            </div>

            <SearchBar onSearch={handleSearch} />
          </div>
        }
      />

      {/* Summary page in new tab */}
      <Route path="/summary/:cnr" element={<SummaryPage />} />
    </Routes>
  );
};

export default App;
