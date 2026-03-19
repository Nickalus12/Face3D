/**
 * Gemini AI Integration for Face3D
 *
 * Provides two capabilities:
 * 1. Quality Critique — analyzes turntable renders to grade reconstruction quality
 * 2. Error Diagnosis — translates pipeline errors into plain-English fixes
 *
 * All calls are optional and gated on having an API key stored in the DB.
 * Uses Gemini 2.0 Flash (free tier, 15 RPM).
 */

import { getSetting, storeAiCritique, setSetting } from './database';

const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models';
const DEFAULT_MODEL = 'gemini-2.0-flash';

// ── API Key Management ─────────────────────────────────────────

export async function getGeminiApiKey(): Promise<string | null> {
  return getSetting('gemini_api_key');
}

export async function setGeminiApiKey(key: string): Promise<void> {
  await setSetting('gemini_api_key', key);
}

export async function clearGeminiApiKey(): Promise<void> {
  const { deleteSetting } = await import('./database');
  await deleteSetting('gemini_api_key');
}

export async function isGeminiEnabled(): Promise<boolean> {
  const enabled = await getSetting('ai_features_enabled');
  if (enabled === 'false') return false;
  const key = await getGeminiApiKey();
  return !!key;
}

export async function setAiFeaturesEnabled(enabled: boolean): Promise<void> {
  await setSetting('ai_features_enabled', enabled ? 'true' : 'false');
}

// ── Core API Call ──────────────────────────────────────────────

interface GeminiPart {
  text?: string;
  inlineData?: {
    mimeType: string;
    data: string; // base64
  };
}

interface GeminiResponse {
  candidates?: {
    content?: {
      parts?: { text?: string }[];
    };
  }[];
  error?: { message: string };
}

async function callGemini(
  parts: GeminiPart[],
  apiKey: string,
  model: string = DEFAULT_MODEL,
): Promise<string> {
  const url = `${GEMINI_API_URL}/${model}:generateContent?key=${apiKey}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts }],
      generationConfig: {
        temperature: 0.3,
        maxOutputTokens: 1024,
      },
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Gemini API error (${response.status}): ${errText}`);
  }

  const data: GeminiResponse = await response.json();
  if (data.error) {
    throw new Error(`Gemini error: ${data.error.message}`);
  }

  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) {
    throw new Error('Gemini returned empty response');
  }
  return text;
}

// ── Quality Critique ───────────────────────────────────────────

export interface QualityCritique {
  grade: string;           // A, B, C, D, F
  score: number;           // 0-100
  summary: string;         // One-line summary
  strengths: string[];     // What's good
  issues: string[];        // What's wrong
  suggestions: string[];   // How to improve
}

/**
 * Analyze turntable renders and grade the reconstruction quality.
 * Sends 4-5 key frames to Gemini Flash for visual analysis.
 *
 * @param imageBase64s Array of base64-encoded PNG images (front, side, back views)
 * @param sessionId Session ID for storing the critique in DB
 * @returns Parsed critique or null if AI is disabled
 */
export async function critiqueRenders(
  imageBase64s: string[],
  sessionId: string,
): Promise<QualityCritique | null> {
  const apiKey = await getGeminiApiKey();
  if (!apiKey) return null;

  const enabled = await isGeminiEnabled();
  if (!enabled) return null;

  const parts: GeminiPart[] = [
    {
      text: `You are a 3D reconstruction quality assessor. Analyze these turntable renders of a 3D Gaussian Splat face reconstruction.

Grade the reconstruction on a scale of A to F:
- A: Photorealistic, no visible artifacts, complete coverage
- B: Good quality, minor artifacts, mostly complete
- C: Acceptable, noticeable artifacts or missing regions
- D: Poor quality, significant issues
- F: Failed reconstruction

Respond in this exact JSON format (no markdown, no code blocks):
{"grade":"B","score":78,"summary":"Good frontal quality with minor ear artifacts","strengths":["Clear facial features","Good skin texture","Accurate nose geometry"],"issues":["Left ear has floater artifacts","Back of head is sparse"],"suggestions":["Capture more side-profile footage","Add 2-3 Expert RAW photos of ears"]}`,
    },
    ...imageBase64s.slice(0, 5).map((b64) => ({
      inlineData: { mimeType: 'image/png', data: b64 },
    })),
  ];

  try {
    const raw = await callGemini(parts, apiKey);
    // Parse JSON from response (strip any markdown wrapping)
    const jsonStr = raw.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    const critique: QualityCritique = JSON.parse(jsonStr);

    // Store in DB
    await storeAiCritique({
      sessionId,
      type: 'quality_critique',
      grade: critique.grade,
      feedback: JSON.stringify(critique),
      model: DEFAULT_MODEL,
    });

    // Update session quality grade
    const { upsertSession } = await import('./database');
    await upsertSession({
      id: sessionId,
      name: sessionId,
      qualityGrade: critique.grade,
      qualityScore: critique.score,
    });

    return critique;
  } catch (e) {
    console.error('[Gemini] Quality critique failed:', e);
    return null;
  }
}

// ── Error Diagnosis ────────────────────────────────────────────

export interface ErrorDiagnosis {
  summary: string;         // Plain English explanation
  cause: string;           // Root cause
  fix: string;             // How to fix it
  severity: 'low' | 'medium' | 'high' | 'critical';
}

/**
 * Translate a pipeline error into a plain-English diagnosis with fix instructions.
 *
 * @param errorLog The raw error text / stack trace
 * @param stageName Which pipeline stage failed
 * @returns Diagnosis or null if AI is disabled
 */
export async function diagnoseError(
  errorLog: string,
  stageName: string,
  sessionId?: string,
): Promise<ErrorDiagnosis | null> {
  const apiKey = await getGeminiApiKey();
  if (!apiKey) return null;

  const enabled = await isGeminiEnabled();
  if (!enabled) return null;

  // Truncate very long error logs
  const truncated = errorLog.length > 3000 ? errorLog.slice(-3000) : errorLog;

  const parts: GeminiPart[] = [
    {
      text: `You are a 3D reconstruction pipeline error diagnostician. A pipeline stage failed. Analyze the error and provide a diagnosis.

Failed Stage: ${stageName}

Error Log:
${truncated}

Respond in this exact JSON format (no markdown, no code blocks):
{"summary":"Your GPU ran out of memory during training","cause":"The model has too many Gaussians (500K) for your 8GB GPU","fix":"Reduce max_num_gaussians to 200000 in config/pipeline.yaml, or reduce the training resolution","severity":"high"}

Severity levels: low (cosmetic), medium (degraded output), high (stage failed), critical (pipeline broken)`,
    },
  ];

  try {
    const raw = await callGemini(parts, apiKey);
    const jsonStr = raw.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    const diagnosis: ErrorDiagnosis = JSON.parse(jsonStr);

    // Store in DB
    if (sessionId) {
      await storeAiCritique({
        sessionId,
        type: 'error_diagnosis',
        grade: diagnosis.severity,
        feedback: JSON.stringify(diagnosis),
        model: DEFAULT_MODEL,
      });
    }

    return diagnosis;
  } catch (e) {
    console.error('[Gemini] Error diagnosis failed:', e);
    return null;
  }
}

// ── Validate API Key ───────────────────────────────────────────

/**
 * Test if an API key is valid by making a minimal request.
 */
export async function validateApiKey(key: string): Promise<{ valid: boolean; error?: string }> {
  try {
    const result = await callGemini(
      [{ text: 'Respond with exactly: OK' }],
      key,
    );
    return { valid: result.includes('OK') };
  } catch (e) {
    return { valid: false, error: e instanceof Error ? e.message : 'Unknown error' };
  }
}
