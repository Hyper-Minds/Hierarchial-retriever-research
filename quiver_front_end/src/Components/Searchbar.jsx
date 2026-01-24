import React, { useState } from "react";

const SearchBar = ({ onSearch }) => {
  const [text, setText] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    onSearch(text);
    setText("");
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="p-4 bg-white border-t flex gap-3"
    >
      <input
        className="flex-1 border rounded px-4 py-2"
        placeholder="Type your query..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button className="bg-black text-white px-6 py-2 rounded">
        Search
      </button>
    </form>
  );
};

export default SearchBar;
