import os
from fastapi import FastAPI, Depends, HTTPException, status, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend import database, auth
from retrieval import retrieve

app = FastAPI(title="NexaCloud Enterprise RAG")

# Create database tables
database.init_db()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def bootstrap_admin():
    db = next(database.get_db())
    admin_user = db.query(database.User).filter(database.User.email == "admin@nexacloud.com").first()
    if not admin_user:
        hashed_pw = auth.get_password_hash("admin123")
        admin = database.User(name="Admin", email="admin@nexacloud.com", password_hash=hashed_pw, role="ADMIN")
        db.add(admin)
        db.commit()

@app.on_event("startup")
def on_startup():
    bootstrap_admin()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    email: str = payload.get("email")
    user = db.query(database.User).filter(database.User.email == email).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

def require_admin(user: database.User = Depends(get_current_user)):
    if user.role != "ADMIN":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user

class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str

class QuestionRequest(BaseModel):
    question: str

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(database.User).filter(database.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    access_token = auth.create_access_token(data={"user_id": user.id, "email": user.email, "role": user.role, "name": user.name})
    return {"access_token": access_token, "token_type": "bearer", "role": user.role, "name": user.name}

@app.post("/rag/ask")
def ask_question(request: QuestionRequest, user: database.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    
    answer = retrieve.get_rag_response(request.question, user.role)
    
    return {"answer": answer}

@app.get("/v1/models")
async def list_models():
    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": "enterprise-rag",
                "object": "model",
                "created": 1677610602,
                "owned_by": "nexacloud"
            }
        ]
    })

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, db: Session = Depends(database.get_db)):
    # 1. Parse Open WebUI request
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    question = messages[-1].get("content", "")
    
    # 2. Extract User Identity
    # Open WebUI sets this header when ENABLE_FORWARD_USER_INFO_HEADERS=true
    user_email = request.headers.get("X-OpenWebUI-User-Email", "unknown@nexacloud.com")
    
    # 3. Lookup Role
    user = db.query(database.User).filter(database.User.email == user_email).first()
    role = user.role if user else "EMPLOYEE"  # Default fallback
    
    print(f"\n[PROXY] Open WebUI Request intercepted for: {user_email} | Role: {role}")
    
    # 4. Perform secure RAG
    answer = retrieve.get_rag_response(question, role)
    
    # 5. Format as OpenAI JSON
    return JSONResponse(content={
        "id": "chatcmpl-proxy123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "enterprise-rag",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer
            },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
    })

@app.post("/users", dependencies=[Depends(require_admin)])
def create_user(user: UserCreate, current_user: database.User = Depends(require_admin), db: Session = Depends(database.get_db)):
    existing = db.query(database.User).filter(database.User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = auth.get_password_hash(user.password)
    db_user = database.User(name=user.name, email=user.email, password_hash=hashed_pw, role=user.role)
    db.add(db_user)
    db.commit()
    return {"message": "User created successfully"}

@app.get("/users/list")
def list_users(db: Session = Depends(database.get_db)):
    users = db.query(database.User).all()
    return [{"name": u.name, "email": u.email, "role": u.role} for u in users]

@app.get("/", response_class=HTMLResponse)
def admin_dashboard():
    html_content = """
    <html>
        <head>
            <title>NexaCloud Admin Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; }
                h1 { color: #003366; }
                .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
                input, select, button { padding: 10px; margin: 5px 0; width: 100%; box-sizing: border-box; }
                button { background-color: #003366; color: white; border: none; cursor: pointer; }
                button:hover { background-color: #002244; }
                table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #003366; color: white; }
            </style>
        </head>
        <body>
            <h1>NexaCloud Enterprise Admin Dashboard</h1>
            
            <div class="card">
                <h2>1. Create User</h2>
                <form id="createUserForm">
                    <input type="text" id="name" placeholder="Name" required>
                    <input type="email" id="email" placeholder="Email (e.g. hr@nexacloud.com)" required>
                    <input type="password" id="password" placeholder="Password" required>
                    <select id="role">
                        <option value="ADMIN">ADMIN</option>
                        <option value="HR_MANAGER">HR_MANAGER</option>
                        <option value="FINANCE_MANAGER">FINANCE_MANAGER</option>
                        <option value="EMPLOYEE">EMPLOYEE</option>
                    </select>
                    <button type="button" onclick="createUser()">Create User</button>
                </form>
                <p id="msg" style="color: green;"></p>
            </div>

            <div class="card">
                <h2>2. System Users</h2>
                <button type="button" onclick="loadUsers()">Refresh Users</button>
                <table id="usersTable">
                    <tr><th>Name</th><th>Email</th><th>Role</th></tr>
                </table>
            </div>

            <script>
                // We mock the token here for simplicity in this localhost dashboard.
                // In production, an Admin would login first.
                // We'll hardcode the admin login to fetch a token for the API calls.
                let adminToken = "";
                
                async function loginAdmin() {
                    let fd = new FormData();
                    fd.append('username', 'admin@nexacloud.com');
                    fd.append('password', 'admin123');
                    let r = await fetch('/login', {method: 'POST', body: fd});
                    if(r.ok) {
                        let data = await r.json();
                        adminToken = data.access_token;
                    }
                }

                async function createUser() {
                    if (!adminToken) await loginAdmin();
                    let payload = {
                        name: document.getElementById("name").value,
                        email: document.getElementById("email").value,
                        password: document.getElementById("password").value,
                        role: document.getElementById("role").value
                    };
                    let r = await fetch('/users', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + adminToken},
                        body: JSON.stringify(payload)
                    });
                    if(r.ok) {
                        document.getElementById("msg").innerText = "User created successfully!";
                        loadUsers();
                    } else {
                        document.getElementById("msg").innerText = "Error creating user.";
                        document.getElementById("msg").style.color = "red";
                    }
                }

                async function loadUsers() {
                    let r = await fetch('/users/list');
                    let users = await r.json();
                    let table = "<tr><th>Name</th><th>Email</th><th>Role</th></tr>";
                    for(let u of users) {
                        table += `<tr><td>${u.name}</td><td>${u.email}</td><td><b>${u.role}</b></td></tr>`;
                    }
                    document.getElementById("usersTable").innerHTML = table;
                }

                window.onload = function() {
                    loadUsers();
                    loginAdmin();
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
