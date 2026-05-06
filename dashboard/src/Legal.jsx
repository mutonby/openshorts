import React from 'react';
import { ArrowLeft } from 'lucide-react';

const LAST_UPDATED = '2026-05-06';
const ISSUES_URL = 'https://github.com/mutonby/openshorts/issues';

function Section({ title, children }) {
    return (
        <section className="mb-7">
            <h2 className="text-lg font-bold text-white mb-2">{title}</h2>
            <div className="text-zinc-300 leading-relaxed space-y-2 text-sm">{children}</div>
        </section>
    );
}

export default function Legal() {
    const handleBack = () => {
        window.location.hash = '';
    };

    return (
        <div className="min-h-screen bg-bg text-white">
            <header className="border-b border-white/5 sticky top-0 bg-bg/95 backdrop-blur z-10">
                <div className="max-w-3xl mx-auto px-6 py-4 flex items-center">
                    <button
                        onClick={handleBack}
                        className="text-zinc-400 hover:text-white flex items-center gap-2 text-sm"
                    >
                        <ArrowLeft size={16} /> Back
                    </button>
                </div>
            </header>

            <main className="max-w-3xl mx-auto px-6 py-12">
                <h1 className="text-3xl md:text-4xl font-bold mb-2">Terms & Privacy</h1>
                <p className="text-zinc-500 text-sm mb-10">Last updated: {LAST_UPDATED}</p>

                <Section title="The short version">
                    <p>
                        OpenShorts is a free, open-source AI clip generator. There are no accounts, no payments, and we
                        do not persistently store the videos you upload or the clips we generate. By using the Service
                        you agree to the points below.
                    </p>
                </Section>

                <Section title="Service is provided as-is">
                    <p>
                        The Service is offered for free, on a best-effort basis, with no warranties of any kind and no
                        guarantee of uptime, accuracy, or fitness for any particular purpose. To the maximum extent
                        permitted by law, we are not liable for any damages arising from your use of the Service.
                    </p>
                </Section>

                <Section title="You are responsible for what you upload">
                    <p>
                        Before processing a video, you must affirmatively confirm — via the checkbox in the upload
                        interface — that you own the content or have the rights to process it. By doing so you
                        represent and warrant that:
                    </p>
                    <ul className="list-disc pl-6 space-y-1">
                        <li>You own all rights to the content, or have a valid license or permission to process it;</li>
                        <li>The content does not infringe any third-party copyright, trademark, privacy, or other right;</li>
                        <li>The content is not unlawful, defamatory, or otherwise prohibited.</li>
                    </ul>
                    <p>
                        If you submit content you do not have rights to, that is your responsibility, not ours. You
                        agree to indemnify OpenShorts and its contributors against any third-party claim arising from
                        content you submitted.
                    </p>
                </Section>

                <Section title="What we keep, and for how long">
                    <ul className="list-disc pl-6 space-y-1">
                        <li>
                            <strong className="text-white">Uploaded videos and generated clips:</strong> deleted with
                            their job, typically within 1 hour. Not backed up off-server in our hosted deployment.
                        </li>
                        <li>
                            <strong className="text-white">Attestation record (IP, user-agent, timestamp, source):</strong>{' '}
                            kept in memory with the job and discarded when the job is purged (≈1 hour). Used only to
                            evidence the ownership confirmation in case of a takedown or dispute.
                        </li>
                        <li>
                            <strong className="text-white">Standard server access logs:</strong> retained up to 30 days
                            for debugging and abuse prevention.
                        </li>
                        <li>
                            <strong className="text-white">API keys (Gemini, ElevenLabs, Upload-Post):</strong> stored
                            encrypted in your browser's <code className="text-zinc-200">localStorage</code>. They are
                            sent as request headers when a feature needs them, used to call the relevant third party,
                            and never written to our database or disk.
                        </li>
                    </ul>
                    <p>We do not sell, rent, or share your data with third parties for advertising or any unrelated purpose.</p>
                </Section>

                <Section title="Third-party APIs">
                    <p>
                        When you use a feature that requires it, OpenShorts forwards relevant data to the third-party
                        API for which you provided a key — Google Gemini (AI analysis), ElevenLabs (optional dubbing),
                        Upload-Post (optional social posting). Those services have their own terms and privacy policies
                        which apply in addition to this notice.
                    </p>
                </Section>

                <Section title="Your rights (EU / EEA / UK)">
                    <p>
                        Under the GDPR / UK GDPR you have the right to access, rectify, erase, restrict, object to, or
                        port your personal data. Because we do not hold accounts and job data is purged within an hour,
                        most requests are auto-satisfied by the retention schedule. For anything else, file a request
                        via{' '}
                        <a className="text-primary underline" href={ISSUES_URL} target="_blank" rel="noopener noreferrer">
                            GitHub Issues
                        </a>
                        . You may also lodge a complaint with your local supervisory authority (in Spain: AEPD,{' '}
                        <a className="text-primary underline" href="https://www.aepd.es" target="_blank" rel="noopener noreferrer">
                            aepd.es
                        </a>
                        ).
                    </p>
                </Section>

                <Section title="Copyright takedowns">
                    <p>
                        If you believe content processed through the Service infringes your copyright, open an issue at{' '}
                        <a className="text-primary underline" href={ISSUES_URL} target="_blank" rel="noopener noreferrer">
                            {ISSUES_URL}
                        </a>{' '}
                        with: identification of the work, identification of the allegedly infringing material (job ID,
                        URL, or sufficient detail to locate it), your contact information, and a statement that you are
                        authorized to act on behalf of the rights holder. Note that uploaded content is typically
                        deleted within 1 hour, so most takedowns are auto-resolved by retention.
                    </p>
                </Section>

                <Section title="Self-hosted instances">
                    <p>
                        OpenShorts is open source and may be self-hosted. This notice applies only to the hosted
                        version we operate. Self-hosted instances are operated by their respective administrators, and
                        their data handling, retention, and policies are their responsibility, not ours.
                    </p>
                </Section>

                <Section title="Changes & contact">
                    <p>
                        We may update this notice from time to time; the "Last updated" date above reflects the most
                        recent revision. Continued use after a change constitutes acceptance. For any other question,
                        please use{' '}
                        <a className="text-primary underline" href={ISSUES_URL} target="_blank" rel="noopener noreferrer">
                            GitHub Issues
                        </a>
                        .
                    </p>
                    <p>This notice is governed by the laws of Spain.</p>
                </Section>
            </main>
        </div>
    );
}
