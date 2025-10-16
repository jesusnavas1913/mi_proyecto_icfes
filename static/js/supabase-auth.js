/**
 * Supabase Authentication Integration
 * Uses Supabase JS client to handle login and registration
 */

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm';

const SUPABASE_URL = 'https://xapvvirzbxydmrymobja.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhhcHZ2aXJ6Ynh5ZG1yeW1vYmphIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg4MTAwMjEsImV4cCI6MjA3NDM4NjAyMX0.1MPy5GIxJOHolm0Xl69bw8TN6h2F1PLaNt-d1CRsIV8';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

/**
 * Login user with Supabase using cedula and password
 * @param {string} cedula
 * @param {string} password
 * @returns {Promise<object>} user session or error
 */
export async function supabaseLogin(cedula, password) {
  // Supabase uses email for login, so we assume cedula is stored as email or use a custom auth flow
  // For this example, we assume cedula is stored in a custom column and we use a RPC or filter to login
  // But Supabase auth only supports email/password by default
  // So we need to query user by cedula and then sign in with email/password

  // Query user by cedula
  const { data: users, error: userError } = await supabase
    .from('usuarios')
    .select('email')
    .eq('cedula', cedula)
    .limit(1)
    .single();

  if (userError || !users) {
    return { error: 'Usuario no encontrado' };
  }

  const email = users.email;

  // Sign in with email and password
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });

  if (error) {
    return { error: error.message };
  }

  return { session: data.session, user: data.user };
}

/**
 * Register user with Supabase
 * @param {object} userData - { nombre, email, cedula, password, rol }
 * @returns {Promise<object>} user or error
 */
export async function supabaseRegister(userData) {
  // Register user with email and password
  const { data, error } = await supabase.auth.signUp({
    email: userData.email,
    password: userData.password,
  });

  if (error) {
    return { error: error.message };
  }

  // Insert additional user data in 'usuarios' table
  const { error: insertError } = await supabase
    .from('usuarios')
    .insert([
      {
        id: data.user.id,
        nombre: userData.nombre,
        email: userData.email,
        cedula: userData.cedula,
        rol: userData.rol || 'estudiante',
      },
    ]);

  if (insertError) {
    return { error: insertError.message };
  }

  return { user: data.user };
}
