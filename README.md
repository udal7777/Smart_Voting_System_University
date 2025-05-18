# 🗳️ Smart Voting System for University

A **Smart Face Detection Based Voting System** developed to conduct secure, transparent, and efficient student elections within a university. This system verifies each voter through facial recognition, ensuring one person gets only one vote and eliminating proxy voting.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Screenshots](#screenshots)
- [Future Scope](#future-scope)
- [Team Members](#team-members)
- [License](#license)

---

## 📖 About the Project

Traditional university voting systems are prone to manipulation and human error. Our project introduces a **face recognition-based smart voting system** that automates the process, enhances security, and makes voting more accessible for students. With face detection, each voter is verified uniquely, making the system reliable and tamper-proof.

---

## ✨ Features

- 🔐 Facial recognition-based authentication
- 🧾 One vote per verified student
- 🗳️ Real-time vote casting
- 📊 Live result updates
- 🖥️ Admin dashboard to manage candidates and votes
- 👨‍🎓 Simple UI for students

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript  
- **Backend**: Python (Flask or Django)  
- **Machine Learning**: OpenCV, face_recognition (dlib)  
- **Database**: SQLite / MySQL / Firebase  
- **Tools**: Git, VS Code

---

## ⚙️ Architecture

Student --> [Face Detection Module] --> [Face Match?]
--> Yes --> [Allow Vote] --> [Store in DB] --> [Show Results]
--> No --> [Access Denied]


---

## 🔄 How It Works

1. **Admin Registration**: Admin uploads eligible voters' face data and candidate list.
2. **Face Detection**: Student stands in front of the webcam for face scan.
3. **Authentication**: System matches the live face with stored data.
4. **Voting**: On successful verification, the student can vote.
5. **Result Display**: Votes are stored in the database and results are updated live.

---

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/udal7777/smart-voting-system.git
cd smart-voting-system

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

