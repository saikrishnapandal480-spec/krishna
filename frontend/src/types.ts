export interface Mission {
  day: number;
  title: string;
  passed?: boolean;
  attempts?: number;
  skipped?: boolean;
}

export interface Member {
  id: string;
  name: string;
  jobRole: string;
  yearsExperience: number;
  education: string;
  status?: string;
}

export interface Candidate {
  member: Member;
  missions: Mission[];
  signals?: {
    commitDays: number;
    missionsCompleted: number;
    missionsFirstTry: number;
  };
}

export interface Feedback {
  summary: string;
  strengths: string[];
  gaps: string[];
  next: string[];
}

export interface InterviewResponse {
  reply: string;
  done: boolean;
  feedback?: Feedback;
  questionNumber?: number;
  currentTopic?: string;
  curriculumDay?: number;
  coveredDaysCount?: number;
  difficulty?: string;
}
