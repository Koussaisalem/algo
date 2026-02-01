#!/usr/bin/env ts-node

/**
 * Database setup script
 * Run with: npm run db:setup
 */

import { initDatabase, userDb } from '../lib/db';
import bcrypt from 'bcryptjs';

async function setupDatabase() {
  console.log('🔧 Setting up QuantumLab database...\n');

  try {
    // Initialize schema
    await initDatabase();
    console.log('✓ Database schema created\n');

    // Check if admin user exists
    const adminEmail = 'admin@quantumlab.com';
    const existingAdmin = await userDb.findByEmail(adminEmail);

    if (!existingAdmin) {
      // Create default admin user
      const adminPassword = await bcrypt.hash('admin123', 12);
      await userDb.create(adminEmail, 'Admin User', adminPassword);
      console.log('✓ Created default admin user');
      console.log(`  Email: ${adminEmail}`);
      console.log(`  Password: admin123`);
      console.log('  ⚠️  Change this password in production!\n');
    } else {
      console.log('✓ Admin user already exists\n');
    }

    // Show user count
    const userCount = await userDb.count();
    console.log(`📊 Total users: ${userCount}\n`);

    console.log('✅ Database setup complete!');
    process.exit(0);
  } catch (error) {
    console.error('❌ Database setup failed:', error);
    process.exit(1);
  }
}

setupDatabase();
