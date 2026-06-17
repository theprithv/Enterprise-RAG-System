import os
import sys
import requests
import getpass

API_URL = "http://localhost:8001"
TOKEN = None
CURRENT_USER = None
CURRENT_ROLE = None

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title="Enterprise RAG Assistant"):
    print("=" * 60)
    print(f" {title:^58}")
    print("=" * 60)

def login():
    global TOKEN, CURRENT_USER, CURRENT_ROLE
    print("\n--- Login ---")
    email = input("Email: ").strip()
    password = input("Password (visible for testing): ").strip()
    
    try:
        response = requests.post(f"{API_URL}/login", data={"username": email, "password": password})
        if response.status_code == 200:
            data = response.json()
            TOKEN = data["access_token"]
            CURRENT_USER = data["name"]
            CURRENT_ROLE = data["role"]
            print("\n[✓] Login successful!")
        else:
            print("\n[✗] Login failed. Please check your credentials.")
    except Exception as e:
        print(f"\n[✗] Could not connect to backend ({API_URL}). Is the FastAPI server running?")

def ask_question():
    global TOKEN
    while True:
        question = input("\nYour question (or type 'back' to go to menu): ").strip()
        if question.lower() == 'back':
            break
        if not question:
            continue
            
        print("\nThinking...")
        try:
            headers = {"Authorization": f"Bearer {TOKEN}"}
            payload = {"question": question}
            response = requests.post(f"{API_URL}/rag/ask", json=payload, headers=headers)
            
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer provided.")
                print("\n" + "═" * 60)
                print(" 🤖 ANSWER:")
                print(" " + "-" * 58)
                print(f" {answer}")
                print("═" * 60)
            elif response.status_code == 401:
                print("\n[✗] Session expired. Please login again.")
                TOKEN = None
                break
            elif response.status_code == 403:
                print("\n[✗] Access denied by RBAC filter.")
            else:
                print(f"\n[✗] Error: {response.text}")
        except Exception as e:
            print(f"\n[✗] Connection error: {e}")

def main():
    global TOKEN, CURRENT_USER, CURRENT_ROLE
    
    while True:
        if not TOKEN:
            clear_screen()
            print_header()
            print("\n1. Login")
            print("2. Exit\n")
            choice = input("Select an option: ").strip()
            
            if choice == "1":
                login()
                if TOKEN:
                    input("\nPress Enter to continue...")
            elif choice == "2":
                print("\nGoodbye.")
                sys.exit(0)
            else:
                print("\nInvalid choice.")
        
        else:
            clear_screen()
            print_header()
            print(f"\nWelcome {CURRENT_USER}")
            print(f"Role: {CURRENT_ROLE}\n")
            print("Options:")
            print("1. Ask Question")
            print("2. Logout\n")
            choice = input("Select an option: ").strip()
            
            if choice == "1":
                ask_question()
            elif choice == "2":
                print("\nLogging out...")
                TOKEN = None
                CURRENT_USER = None
                CURRENT_ROLE = None
            else:
                print("\nInvalid choice.")

if __name__ == "__main__":
    main()
