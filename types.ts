export interface User {
  email: string;
  name: string;
  role: 'user' | 'admin';
  password?: string;
}

export interface Job {
  job_id: string;
  user: {
    email: string;
    name: string;
  };
  mode_used: string;
  status: 'production_ready' | 'needs_review' | string;
  fidelity: number;
  best_candidate: string;
  selection_reason: string;
  warnings: string[];
  downloads: Record<string, string>;
}
