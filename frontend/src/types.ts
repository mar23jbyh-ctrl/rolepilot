export type Role = "assistant" | "user" | "system";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface QuestionDetail {
  question_id: number;
  question?: string;
  question_summary?: string;
  score?: number;
  max_score?: number;
  is_follow_up?: boolean;
  covered_points?: string[];
  missed_points?: string[];
  dimensions?: Record<string, number>;
}

export interface Report {
  overall_score?: number;
  grade?: string;
  text_analysis?: string;
  strengths?: string[];
  weaknesses?: string[];
  per_question?: QuestionDetail[];
  learning_path?: string[];
  token_totals?: { total?: number | null; total_input?: number | null; total_output?: number | null; known_total?: number; usage_complete?: boolean; known_cost_estimate?: number | null };
  cost?: number | null;
}

export interface SessionPayload {
  session_id: string;
  name: string;
  status: string;
  done: boolean;
  question: string;
  question_version: number;
  replayed?: boolean;
  processing?: boolean;
  messages: ChatMessage[];
  report: Report | null;
  usage_total: number | null;
  usage_summary?: { usage_complete: boolean; known_total: number; unknown_calls: number };
  // M23：系统识别到的岗位与题量（用于用户核对）
  job_title?: string;
  domain?: string;
  industry?: string;
  plan_question_count?: number;
  effective_sample_count?: number;
}

export interface AnswerRequest {
  answer: string;
  answer_request_id: string;
  expected_question_version: number;
}

/** A 202 is not a new question or a completed SessionPayload. */
export interface AnswerProcessing {
  processing: boolean;
  session_id: string;
  answer_request_id?: string;
  status?: string;
}

export type AnswerResult = SessionPayload | AnswerProcessing;

export interface SessionSummary {
  id: string;
  name: string;
  status: string;
  total_tokens: number | null;
  created_at: string;
  updated_at: string;
}

export interface UploadResult {
  text: string;
  method: string;
  file_name: string;
}
