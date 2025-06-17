# Question Deck: Portuguese Personalized Question Generator

An interactive AI-powered app that generates introspective and emotionally engaging questions to strengthen the bond between two people — whether it's friendship or a romantic relationship.

## Features

- Personalized question generation based on:
  - Type of relationship (friendship or love)
  - Age, gender, profession, and shared interests
  - Relationship duration and goals
- Interactive chat interface using Gradio
- Emotional connection evaluation based on user responses
- Contextual knowledge extracted from relevant psychology/self-help books

## Technologies Used

- **Python 3.10+**
- **Gradio** – chatbot interface
- **LangChain + ChromaDB** – PDF vector storage
- **OpenAI (GPT-4o-mini)** – question generation & relationship evaluation
- **Docker + Docker Compose** – containerization and deployment
- **dotenv** – environment variable management

---

## Getting Started

### 1. Clone the project

```bash
git clone https://github.com/bernardinosousa/connections-chatbot.git
cd connections-chatbot
```

### 2. Add environment variables

Copy `.env.example` to `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-...
```

### 3. Add your PDF content (Optional)

Place your PDFs inside the folders:

```
content/
├── friendship/
│   └── [PDFs about friendship]
└── love/
    └── [PDFs about romantic relationships]
```

### 4. Build and run with Docker Compose

```bash
docker-compose up --build
```

Access the app at [http://localhost:7860](http://localhost:7860)

---

## Quick Test

When prompted:

1. Enter "friendship" or "love"
2. Follow the questions to input both people's info
3. Respond to the AI-generated questions
4. Receive a connection evaluation score at the end

---

## License

MIT License – free to use, modify, and share.

---

## 🙋‍♂️ Contributing

Suggestions and PRs are welcome! Let’s make meaningful conversations happen.
