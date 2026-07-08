import fs from 'fs';
import path from 'path';
import bcrypt from 'bcryptjs';
import { User } from './types.js';

const DATA_ROOT = process.env.VEKTORYUM_DATA_ROOT || './vektoryum_data';
const USERS_FILE = path.join(DATA_ROOT, 'users.json');

// In-memory sessions store
export const SESSIONS: Map<string, { email: string }> = new Map();

export function hashPassword(password: string): string {
  return bcrypt.hashSync(password, 10);
}

export function verifyPassword(password: string, hash: string): boolean {
  try {
    return bcrypt.compareSync(password, hash);
  } catch (err) {
    return false;
  }
}

export function loadUsers(): Record<string, User> {
  if (!fs.existsSync(DATA_ROOT)) {
    fs.mkdirSync(DATA_ROOT, { recursive: true });
  }

  if (!fs.existsSync(USERS_FILE)) {
    const adminEmail = (process.env.VEKTORYUM_ADMIN_EMAIL || 'admin@vektoryum.local').toLowerCase().trim();
    const adminPassword = process.env.VEKTORYUM_ADMIN_PASSWORD || 'admin123';
    
    const initialUsers: Record<string, User> = {
      [adminEmail]: {
        email: adminEmail,
        name: 'Vektoryum Yönetici',
        role: 'admin',
        password: hashPassword(adminPassword),
      },
    };
    fs.writeFileSync(USERS_FILE, JSON.stringify(initialUsers, null, 2), 'utf-8');
  }

  try {
    const content = fs.readFileSync(USERS_FILE, 'utf-8');
    return JSON.parse(content);
  } catch (err) {
    return {};
  }
}

export function saveUsers(users: Record<string, User>): void {
  if (!fs.existsSync(DATA_ROOT)) {
    fs.mkdirSync(DATA_ROOT, { recursive: true });
  }
  fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2), 'utf-8');
}

export function safeUser(user: User): Partial<User> {
  return {
    email: user.email,
    name: user.name,
    role: user.role,
  };
}

export function getCurrentUser(sessionToken: string | undefined): User | null {
  if (!sessionToken) return null;
  const session = SESSIONS.get(sessionToken);
  if (!session) return null;
  
  const users = loadUsers();
  const user = users[session.email.toLowerCase()];
  return user || null;
}
