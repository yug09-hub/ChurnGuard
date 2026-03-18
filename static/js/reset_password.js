/**
 * ChurnGuard - Reset Password JavaScript
 */

const resetPasswordForm = document.getElementById('resetPasswordForm');
const newPasswordInput = document.getElementById('newPassword');
const confirmPasswordInput = document.getElementById('confirmPassword');
const resetTokenInput = document.getElementById('resetToken');
const resetBtn = document.getElementById('resetBtn');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const toastContainer = document.getElementById('toastContainer');

document.addEventListener('DOMContentLoaded', () => {
    if (resetPasswordForm) {
        resetPasswordForm.addEventListener('submit', handleResetPassword);
    }
    
    // Hide error when typing
    [newPasswordInput, confirmPasswordInput].forEach(input => {
        if (input) {
            input.addEventListener('input', () => {
                errorMessage.style.display = 'none';
            });
        }
    });
});

async function handleResetPassword(e) {
    e.preventDefault();
    
    const password = newPasswordInput.value;
    const confirm = confirmPasswordInput.value;
    const token = resetTokenInput.value;
    
    if (password.length < 8) {
        showError('Password must be at least 8 characters long');
        return;
    }
    
    if (password !== confirm) {
        showError('Passwords do not match');
        return;
    }
    
    setLoading(true);
    
    try {
        const response = await fetch('/api/reset-password', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token, password })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            showToast('Password updated successfully! Redirecting...', 'success');
            setTimeout(() => {
                window.location.href = '/login';
            }, 2000);
        } else {
            showError(data.message || 'Failed to reset password');
        }
    } catch (error) {
        console.error('Reset error:', error);
        showError('Network error. Please try again.');
    } finally {
        setLoading(false);
    }
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    
    // Shake animation
    const card = document.querySelector('.login-card');
    card.style.animation = 'none';
    setTimeout(() => {
        card.style.animation = 'shake 0.5s ease';
    }, 10);
}

function setLoading(loading) {
    resetBtn.disabled = loading;
    resetBtn.classList.toggle('loading', loading);
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
