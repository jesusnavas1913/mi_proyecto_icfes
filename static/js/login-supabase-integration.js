/**
 * This script integrates Supabase authentication with the existing login form.
 * It uses the provided Supabase project URL and Anon Key.
 * It supports login by cedula by querying the 'usuarios' table to get the email,
 * then authenticates with Supabase using email and password.
 */

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm';

const SUPABASE_URL = 'https://xapvvirzbxydmrymobja.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhhcHZ2aXJ6Ynh5ZG1yeW1vYmphIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg4MTAwMjEsImV4cCI6MjA3NDM4NjAyMX0.1MPy5GIxJOHolm0Xl69bw8TN6h2F1PLaNt-d1CRsIV8';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function supabaseLogin(cedula, password) {
  // Query user by cedula to get email
  const { data: user, error } = await supabase
    .from('usuarios')
    .select('email, nombre, rol')
    .eq('cedula', cedula)
    .single();

  if (error || !user) {
    throw new Error('Cédula no encontrada');
  }

  // Sign in with email and password
  const { data, error: signInError } = await supabase.auth.signInWithPassword({
    email: user.email,
    password,
  });

  if (signInError) {
    throw new Error('Contraseña incorrecta');
  }

  return { session: data.session, user: { ...user, id: data.user.id } };
}

// Attach event listener to login form
document.addEventListener('DOMContentLoaded', () => {
  // Hide loading overlay on page load
  const loadingOverlay = document.getElementById('loadingOverlay');
  if (loadingOverlay) {
    loadingOverlay.style.display = 'none';
  }

  const loginForm = document.getElementById('loginForm');
  const alertAuth = document.getElementById('alertAuth');

  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const cedula = document.getElementById('loginCedula').value.trim();
    const password = document.getElementById('loginPassword').value;

    try {
      const { session, user } = await supabaseLogin(cedula, password);
      // Save session info in localStorage
      localStorage.setItem('currentUser', JSON.stringify({ cedula, role: user.rol, name: user.nombre, id: user.id }));

      // Redirect based on role
      if (user.rol === 'estudiante') {
        window.location.href = '/dashboard';
      } else if (user.rol === 'profesor') {
        window.location.href = '/pdf-evaluation';
      } else {
        alertAuth.textContent = 'Rol de usuario desconocido';
        alertAuth.className = 'alert error active';
      }
    } catch (error) {
      alertAuth.textContent = error.message;
      alertAuth.className = 'alert error active';
      setTimeout(() => alertAuth.classList.remove('active'), 4000);
    }
  });
});
