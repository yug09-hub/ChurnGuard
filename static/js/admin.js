/**
 * ChurnGuard - Admin Login JavaScript
 * ======================================
 * Handles login authentication, password visibility toggle,
 * remember me functionality, and theme switching.
 */

// Admin credentials (in production, this should be server-side only)
const ADMIN_USERNAME = 'admin';
const ADMIN_PASSWORD = 'admin123';

// DOM Elements
const loginForm = document.getElementById('loginForm');
const usernameInput = document.getElementById('username');
const passwordInput = document.getElementById('password');
const passwordToggle = document.getElementById('passwordToggle');
const passwordIcon = document.getElementById('passwordIcon');
const rememberMeCheckbox = document.getElementById('rememberMe');
const loginBtn = document.getElementById('loginBtn');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');
const toastContainer = document.getElementById('toastContainer');

// Forgot Password specific elements
const forgotPasswordBtn = document.getElementById('forgotPasswordBtn');
const backToLoginBtn = document.getElementById('backToLoginBtn');
const forgotPasswordForm = document.getElementById('forgotPasswordForm');
const resetUsernameInput = document.getElementById('resetUsername');
const resetEmailInput = document.getElementById('resetEmail');
const sendResetBtn = document.getElementById('sendResetBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    checkExistingSession();
    loadRememberedCredentials();
    setupEventListeners();
    animateEntrance();
});

// Theme Management
function initTheme() {
    const savedTheme = localStorage.getItem('admin-theme') || 'dark';
    setTheme(savedTheme);
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('admin-theme', theme);
    
    if (themeIcon) {
        themeIcon.className = theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

// Check for existing session
function checkExistingSession() {
    const sessionToken = sessionStorage.getItem('admin-session');
    if (sessionToken) {
        // Verify session with server
        verifySession(sessionToken);
    }
}

async function verifySession(token) {
    try {
        const response = await fetch('/api/verify-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token })
        });
        
        if (response.ok) {
            window.location.href = '/';
        } else {
            sessionStorage.removeItem('admin-session');
        }
    } catch (error) {
        console.error('Session verification failed:', error);
        sessionStorage.removeItem('admin-session');
    }
}

// Load remembered credentials
function loadRememberedCredentials() {
    const remembered = localStorage.getItem('admin-remember');
    if (remembered) {
        try {
            const data = JSON.parse(remembered);
            usernameInput.value = data.username || '';
            rememberMeCheckbox.checked = true;
        } catch (e) {
            console.error('Failed to parse remembered credentials');
        }
    }
}

// Save remembered credentials
function saveRememberedCredentials(username) {
    if (rememberMeCheckbox.checked) {
        localStorage.setItem('admin-remember', JSON.stringify({ username }));
    } else {
        localStorage.removeItem('admin-remember');
    }
}

// Event Listeners
function setupEventListeners() {
    // Theme toggle
    themeToggle.addEventListener('click', toggleTheme);
    
    // Password visibility toggle
    passwordToggle.addEventListener('click', togglePasswordVisibility);
    
    // Form submission
    loginForm.addEventListener('submit', handleLogin);
    
    // Input focus effects
    usernameInput.addEventListener('input', () => hideError());
    passwordInput.addEventListener('input', () => hideError());
    
    // Password input validation - check minimum length
    passwordInput.addEventListener('input', validatePasswordLength);
    
    // Enter key on password field
    passwordInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            loginForm.dispatchEvent(new Event('submit'));
        }
    });

    // Forgot password flow
    const loginHeader = document.getElementById('loginHeader');
    if (forgotPasswordBtn) {
        forgotPasswordBtn.addEventListener('click', () => {
            if (loginHeader) loginHeader.style.display = 'none';
            loginForm.style.display = 'none';
            forgotPasswordForm.style.display = 'flex';
            hideError();
        });
    }

    if (backToLoginBtn) {
        backToLoginBtn.addEventListener('click', () => {
            if (loginHeader) loginHeader.style.display = 'block';
            forgotPasswordForm.style.display = 'none';
            loginForm.style.display = 'flex';
            hideError();
        });
    }

    if (forgotPasswordForm) {
        forgotPasswordForm.addEventListener('submit', handleForgotPassword);
    }
}

// Handle Forgot Password
async function handleForgotPassword(e) {
    e.preventDefault();
    
    const username = resetUsernameInput.value.trim();
    const email = resetEmailInput.value.trim();
    
    if (!username || !email) {
        showError('Please enter both username and email');
        return;
    }
    
    setResetLoading(true);
    hideError();
    
    try {
        const response = await fetch('/api/forgot-password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            showToast(data.message, 'success');
            // Reset form and go back to login after success
            setTimeout(() => {
                forgotPasswordForm.reset();
                backToLoginBtn.click();
            }, 3000);
        } else {
            showError(data.message || 'Error occurred. Please try again.');
            shakeForm();
        }
    } catch (error) {
        console.error('Forgot password error:', error);
        showError('Network error. Please try again.');
        shakeForm();
    } finally {
        setResetLoading(false);
    }
}

function setResetLoading(loading) {
    if (sendResetBtn) {
        sendResetBtn.disabled = loading;
        sendResetBtn.classList.toggle('loading', loading);
    }
}

// Validate password length and update UI
function validatePasswordLength() {
    const password = passwordInput.value;
    const minLength = 8;
    
    if (password.length > 0 && password.length < minLength) {
        showPasswordError(`Password must be at least ${minLength} characters`);
        loginBtn.disabled = true;
    } else {
        hidePasswordError();
        loginBtn.disabled = false;
    }
}

// Show password validation error below password field
function showPasswordError(message) {
    // Remove existing error if any
    hidePasswordError();
    
    const errorDiv = document.createElement('div');
    errorDiv.id = 'passwordValidationError';
    errorDiv.style.color = '#ef4444';
    errorDiv.style.fontSize = '12px';
    errorDiv.style.marginTop = '6px';
    errorDiv.style.display = 'flex';
    errorDiv.style.alignItems = 'center';
    errorDiv.style.gap = '6px';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    
    passwordInput.parentElement.parentElement.appendChild(errorDiv);
}

// Hide password validation error
function hidePasswordError() {
    const existingError = document.getElementById('passwordValidationError');
    if (existingError) {
        existingError.remove();
    }
}

// Toggle password visibility
function togglePasswordVisibility() {
    const isPassword = passwordInput.type === 'password';
    passwordInput.type = isPassword ? 'text' : 'password';
    passwordIcon.className = isPassword ? 'fas fa-eye-slash' : 'fas fa-eye';
    
    // Add a subtle animation
    passwordToggle.style.transform = 'translateY(-50%) scale(0.9)';
    setTimeout(() => {
        passwordToggle.style.transform = 'translateY(-50%) scale(1)';
    }, 150);
}

// Handle login
async function handleLogin(e) {
    e.preventDefault();
    
    const username = usernameInput.value.trim();
    const password = passwordInput.value;
    const minLength = 8;
    
    // Validation
    if (!username || !password) {
        showError('Please enter both username and password');
        shakeForm();
        return;
    }
    
    // Check password length
    if (password.length < minLength) {
        showError(`Password must be at least ${minLength} characters`);
        shakeForm();
        passwordInput.focus();
        return;
    }
    
    // Show loading state
    setLoading(true);
    hideError();
    
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Save session
            sessionStorage.setItem('admin-session', data.token);
            
            // Save remembered credentials
            saveRememberedCredentials(username);
            
            // Show success toast
            showToast('Login successful! Redirecting...', 'success');
            
            // Redirect to dashboard
            setTimeout(() => {
                window.location.href = '/';
            }, 800);
        } else {
            // Show specific error for wrong password
            console.log('Login failed with status:', response.status);
            if (response.status === 401) {
                showError('Wrong password. Please try again.');
                console.log('Showing wrong password message');
            } else {
                showError(data.message || 'Invalid credentials');
            }
            shakeForm();
            passwordInput.value = '';
            passwordInput.focus();
            // Re-validate to disable button
            validatePasswordLength();
        }
    } catch (error) {
        console.error('Login error:', error);
        showError('Network error. Please try again.');
        shakeForm();
    } finally {
        setLoading(false);
    }
}

// UI Helpers
function showError(message) {
    if (errorText) errorText.textContent = message;
    if (errorMessage) {
        errorMessage.style.display = 'flex';
        errorMessage.classList.add('show');
    }
}

function hideError() {
    if (errorMessage) {
        errorMessage.style.display = 'none';
        errorMessage.classList.remove('show');
    }
}

function setLoading(loading) {
    loginBtn.disabled = loading;
    loginBtn.classList.toggle('loading', loading);
}

function shakeForm() {
    const card = document.querySelector('.login-card');
    card.style.animation = 'none';
    setTimeout(() => {
        card.style.animation = 'shake 0.5s ease';
    }, 10);
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
        <span>${message}</span>
    `;
    
    toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-20px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Entrance animation
function animateEntrance() {
    const elements = document.querySelectorAll('.logo-section, .login-card, .login-footer');
    elements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        setTimeout(() => {
            el.style.transition = 'all 0.6s ease';
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

// Security: Clear sensitive data on page unload
window.addEventListener('beforeunload', () => {
    // Don't clear if remember me is checked
    if (!rememberMeCheckbox.checked) {
        usernameInput.value = '';
    }
    passwordInput.value = '';
});

// Prevent going back to login page after successful login
window.addEventListener('pageshow', (e) => {
    if (e.persisted && sessionStorage.getItem('admin-session')) {
        window.location.href = '/';
    }
});
