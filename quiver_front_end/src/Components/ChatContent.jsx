import React from "react";
import ReactMarkdown from "react-markdown";

const ChatContent = ({ messages }) => {
  return (
    <div className="h-full overflow-y-auto p-6 space-y-4">
      {messages.map((msg, index) => (
        <div
          key={index}
          className={`max-w-xl p-4 rounded-lg ${
            msg.role === "user"
              ? "bg-blue-100 ml-auto"
              : "bg-white mr-auto"
          }`}
        >
          <p className="text-gray-800 whitespace-pre-wrap">
            <ReactMarkdown>
            {msg.text}
            </ReactMarkdown>
          </p>
        </div>
      ))}
    </div>
  );
};

export default ChatContent;
