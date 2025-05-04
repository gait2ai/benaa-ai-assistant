// script.js
document.addEventListener('DOMContentLoaded', function () {
    const chatMessages = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');

    // API URL based on current origin
    const API_URL = window.location.origin + "/api/chat";

    // Store messages in sessionStorage
    let chatHistory = [];

    // Load chat history from sessionStorage on page load
    function loadChatHistory() {
        const storedHistory = sessionStorage.getItem('chatHistory');
        if (storedHistory) {
            chatHistory = JSON.parse(storedHistory);
            chatHistory.forEach(item => {
                if (item.type === 'user') {
                    addUserMessage(item.text, item.timestamp, false);
                } else {
                    addAssistantMessage(item.text, item.timestamp, false);
                }
            });
        }
    }

    // Create typing indicator element
    function createTypingIndicator() {
        const div = document.createElement('div');
        div.className = 'message assistant-message typing-indicator';
        
        const logo = document.createElement('img');
        logo.src = "logo.png";
        logo.alt = "شعار الجمعية";
        logo.className = 'assistant-logo';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = "جاري الكتابة <div class='loading-dots'><span></span><span></span><span></span></div>";
        
        div.appendChild(logo);
        div.appendChild(content);
        return div;
    }

    // Format timestamp in Arabic
    function formatTimestamp() {
        const now = new Date();
        
        // Format time as HH:MM in Arabic
        const timeOptions = { hour: '2-digit', minute: '2-digit', hour12: false };
        const time = now.toLocaleTimeString('ar-SA', timeOptions);
        
        // Format date as YYYY-MM-DD in Arabic
        const dateOptions = { year: 'numeric', month: '2-digit', day: '2-digit' };
        const date = now.toLocaleDateString('ar-SA', dateOptions);
        
        return `${time} | ${date}`;
    }

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

    function addUserMessage(text, timestamp = null, saveToHistory = true) {
        const div = document.createElement('div');
        div.className = 'message user-message';
        
        const content = document.createElement('div');
        content.textContent = text;
        div.appendChild(content);
        
        // Add timestamp
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = timestamp || formatTimestamp();
        div.appendChild(timeDiv);
        
        chatMessages.appendChild(div);
        
        // Save to history if needed
        if (saveToHistory) {
            const currentTimestamp = formatTimestamp();
            chatHistory.push({
                type: 'user',
                text: text,
                timestamp: currentTimestamp
            });
            sessionStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        }
        
        scrollToBottom();
    }

    function addAssistantMessage(text, timestamp = null, saveToHistory = true) {
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
        
        // Add timestamp
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = timestamp || formatTimestamp();
        div.appendChild(timeDiv);
        
        chatMessages.appendChild(div);
        
        // Save to history if needed
        if (saveToHistory && !text.includes('loading-dots')) {
            const currentTimestamp = formatTimestamp();
            chatHistory.push({
                type: 'assistant',
                text: text,
                timestamp: currentTimestamp
            });
            sessionStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        }
        
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

        // Add typing indicator
        const typingIndicator = createTypingIndicator();
        chatMessages.appendChild(typingIndicator);
        scrollToBottom();

        try {
            const res = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });

            try {
                // Remove typing indicator
                typingIndicator.remove();
                
                const data = await res.json();
                addAssistantMessage(data.answer);
            } catch (jsonError) {
                console.error("JSON parsing error:", jsonError);
                // Remove typing indicator
                typingIndicator.remove();
                addAssistantMessage("حدث خطأ غير متوقع في قراءة الرد.");
            }
        } catch (err) {
            console.error(err);
            // Remove typing indicator
            typingIndicator.remove();
            addAssistantMessage("حدث خطأ أثناء الاتصال بالخادم.");
        }
    }
    
    // Load chat history when page loads
    loadChatHistory();
});