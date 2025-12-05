import pg from 'pg'
import 'dotenv/config'

const pool = new pg.Pool({
  connectionString: process.env.URL_DB,
})

export const query = (text, params) => pool.query(text, params)

export async function initDb() {
  const createUsageTable = `
    CREATE TABLE IF NOT EXISTS openai_usage (
      id SERIAL PRIMARY KEY,
      timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      model TEXT,
      prompt_tokens INTEGER,
      completion_tokens INTEGER,
      total_tokens INTEGER,
      mastertext TEXT
    );
  `

  const createDailySummaryTable = `
    CREATE TABLE IF NOT EXISTS daily_usage_summary (
      date DATE PRIMARY KEY,
      total_tokens INTEGER DEFAULT 0,
      total_cost NUMERIC(10, 6) DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS message_logs (
      id SERIAL PRIMARY KEY,
      timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
      type TEXT,
      input_content TEXT,
      output_content TEXT
    );
  `

  try {
    await query(createUsageTable)
    await query(createDailySummaryTable)
    
    // Add columns to message_logs if they don't exist
    await query('ALTER TABLE message_logs ADD COLUMN IF NOT EXISTS tokens INTEGER DEFAULT 0')
    await query('ALTER TABLE message_logs ADD COLUMN IF NOT EXISTS cost NUMERIC(10, 6) DEFAULT 0')

    console.log('Database tables initialized')
  } catch (err) {
    console.error('Error initializing database', err)
  }
}
