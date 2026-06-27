export default function ChatUserQuestion({ question }) {
  return (
    <section className="chat-user-question">
      <span>User question</span>
      <p>{question}</p>
    </section>
  );
}