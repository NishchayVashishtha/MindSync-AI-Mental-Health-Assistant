// Authentication functions for MindSync

// Function to handle user registration
function registerUser(event) {
  event.preventDefault();
  
  // Get form values
  const name = document.getElementById('fullName').value;
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;
  const confirmPassword = document.getElementById('confirmPassword').value;
  
  // Validate passwords match
  if (password !== confirmPassword) {
    alert("Passwords don't match!");
    return false;
  }
  
  // Check if email already exists
  const users = JSON.parse(localStorage.getItem('mindsync_users') || '[]');
  if (users.some(user => user.email === email)) {
    alert("This email is already registered. Please use a different email or login.");
    return false;
  }
  
  // Create user object
  const newUser = {
    name: name,
    email: email,
    password: password // In a real app, you would hash this password
  };
  
  // Add to users array
  users.push(newUser);
  
  // Save to localStorage
  localStorage.setItem('mindsync_users', JSON.stringify(users));
  
  // Redirect to login page
  alert("Registration successful! Please login with your credentials.");
  window.location.href = 'login.html';
  return true;
}

// Function to handle user login
function loginUser(event) {
  event.preventDefault();
  
  // Get form values
  const email = document.getElementById('loginEmail').value;
  const password = document.getElementById('loginPassword').value;
  
  // Get users from localStorage
  const users = JSON.parse(localStorage.getItem('mindsync_users') || '[]');
  
  // Find user with matching email and password
  const user = users.find(user => user.email === email && user.password === password);
  
  if (user) {
    // Store current user info (don't store password in session)
    const currentUser = {
      name: user.name,
      email: user.email,
      isLoggedIn: true
    };
    
    // Save current user to session
    sessionStorage.setItem('mindsync_current_user', JSON.stringify(currentUser));
    
    // Redirect to dashboard
    window.location.href = 'dashboard.html';
    return true;
  } else {
    alert("Invalid email or password. Please try again.");
    return false;
  }
}

// Function to check if user is logged in
function checkLoginStatus() {
  const currentUser = JSON.parse(sessionStorage.getItem('mindsync_current_user') || '{}');
  return currentUser.isLoggedIn === true;
}

// Function to logout user
function logoutUser() {
  sessionStorage.removeItem('mindsync_current_user');
  window.location.href = 'index.html';
}