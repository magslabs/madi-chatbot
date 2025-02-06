import { useState } from "react";
import { Send } from "lucide-react";

function App() {
  const [messages, setMessages] = useState([
    { text: "Hello! How can I assist you today?", sender: "bot" }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    setMessages([...messages, { text: input, sender: "user" }]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/chatbot/prompt", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ prompt: input })
      });

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        { text: data.response, sender: "bot" }
      ]);
    } catch (error) {
      console.error("Error fetcthing response:", error);
      setMessages((prev) => [
        ...prev,
        { text: "Sorry, I don't understand your question.", sender: "bot" }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100 p-28">
      <div className="flex flex-col h-full w-full bg-white shadow-lg rounded-lg">
        <header className="bg-pink-500 text-white p-4 text-center font-bold rounded-t-lg">
          MADI Chatbot
        </header>
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`max-w-xs p-4 rounded-lg ${
                msg.sender === "user"
                  ? "bg-blue-500 text-white self-end ml-auto"
                  : "bg-gray-300 text-black self-start"
              }`}
            >
              {msg.text}
            </div>
          ))}
          {isTyping && (
            <div className="max-w-xs p-3 rounded-lg bg-gray-300 text-black self-start">
              MADI is typing...
            </div>
          )}
        </div>
        <div className="p-4 border-t bg-white flex items-center gap-2 rounded-b-lg">
          <input
            className="flex-1 p-2 border rounded-lg"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button onClick={sendMessage} className="p-2 bg-pink-500 text-white rounded-lg">
            <Send className="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>
  );
}

export default App
