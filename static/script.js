// script.js
document.addEventListener('DOMContentLoaded', function () {
    const chatMessages = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');

    // API URL based on current origin
    const API_URL = window.location.origin + "/api/chat";

    // Event listeners for quick question buttons
    document.querySelectorAll('.quick-btn').forEach(button => {
        button.addEventListener('click', () => {
            messageInput.value = button.textContent;
            sendMessage();
        });
    });

    // Event listeners for sending messages
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function addUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'message user-message';
        div.textContent = text;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function addAssistantMessage(text) {
        const div = document.createElement('div');
        div.className = 'message assistant-message';

        const logo = document.createElement('img');
        logo.src = "logo.png";
        logo.alt = "شعار الجمعية";
        logo.className = 'assistant-logo';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = text;

        div.appendChild(logo);
        div.appendChild(content);
        chatMessages.appendChild(div);
        scrollToBottom();
        return div;
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        addUserMessage(message);
        messageInput.value = '';

        const loadingMsg = addAssistantMessage("جاري الكتابة <div class='loading-dots'><span></span><span></span><span></span></div>");

        try {
            const res = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });

            try {
                const data = await res.json();
                chatMessages.lastChild.remove();
                addAssistantMessage(data.answer);
            } catch (jsonError) {
                console.error("JSON parsing error:", jsonError);
                chatMessages.lastChild.remove();
                addAssistantMessage("حدث خطأ غير متوقع في قراءة الرد.");
            }
        } catch (err) {
            console.error(err);
            chatMessages.lastChild.remove();
            addAssistantMessage("حدث خطأ أثناء الاتصال بالخادم.");
        }
    }
});