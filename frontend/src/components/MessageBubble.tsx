import type { ChatMessage } from "../types";

interface BubbleProps {
  message: ChatMessage;
  index: number;
}

export function MessageBubble({ message, index }: BubbleProps) {
  const isUser = message.role === "user";
  const role = isUser ? "user" : "interviewer";
  return (
    <div className={`chat-message ${role}`} data-message-index={index}>
      <div className="chat-avatar" aria-hidden="true">
        {isUser ? "我" : "面试官"}
      </div>
      <div className={`chat-bubble ${role}`}>{message.content}</div>
    </div>
  );
}

export function EvaluationBubble({ text }: { text: string }) {
  return (
    <div className="chat-message interviewer">
      <div className="chat-avatar chat-avatar--system" aria-hidden="true">
        面试官
      </div>
      <div className="chat-bubble system">{text}</div>
    </div>
  );
}
