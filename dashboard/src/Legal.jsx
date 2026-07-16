import React from 'react';
import { ArrowLeft } from 'lucide-react';

const LAST_UPDATED = '2026-07-15';
const ISSUES_URL = 'https://github.com/mutonby/openshorts/issues';
const SUPPORT_EMAIL = 'info@openshorts.app';

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
                        OpenShorts is an AI clip generator. There are two ways to use it:
                    </p>
                    <ul className="list-disc pl-6 space-y-1">
                        <li>
                            <strong className="text-white">Self-hosted (free):</strong> the open-source software, run on
                            your own machine with your own API keys. No account, no payment, no data held by us.
                        </li>
                        <li>
                            <strong className="text-white">Hosted at openshorts.app (paid):</strong> we run everything for
                            you. It requires an account and a paid subscription (with a free trial), and we store the
                            videos you generate while your subscription is active.
                        </li>
                    </ul>
                    <p>By using the hosted Service you agree to the terms below.</p>
                </Section>

                <Section title="Accounts & sign-in">
                    <p>
                        The hosted Service requires an account. You can sign in with a magic link sent to your email or
                        with Google. We store your email address to operate your account and authenticate you. You are
                        responsible for keeping access to your inbox / Google account secure.
                    </p>
                </Section>

                <Section title="Free trial, plans & billing">
                    <p>
                        Paid plans are <strong className="text-white">Starter ($12/mo · 100 min)</strong>,{' '}
                        <strong className="text-white">Creator ($29/mo · 300 min)</strong> and{' '}
                        <strong className="text-white">Pro ($59/mo · 750 min)</strong>, each also available annually (two
                        months free). "Minutes" are minutes of input video processed per billing period; additional
                        minutes can be bought as one-off top-ups.
                    </p>
                    <p>
                        <strong className="text-white">Free trial:</strong> new subscriptions start with a{' '}
                        <strong className="text-white">3-day free trial</strong> that includes up to{' '}
                        <strong className="text-white">20 minutes</strong> of video processing. You provide a payment
                        method up front but are <strong className="text-white">not charged during the trial</strong>. If
                        you do not cancel before the trial ends, your subscription automatically begins, your full plan
                        minutes unlock, and your payment method is charged the plan price. You can cancel at any time from
                        your account.
                    </p>
                    <p>
                        <strong className="text-white">Auto-renewal:</strong> subscriptions renew automatically each
                        period (monthly or yearly) at the then-current price until you cancel. We'll email a reminder
                        before the trial converts to a paid subscription.
                    </p>
                    <p>
                        Payments are processed by <strong className="text-white">Stripe</strong>. We never see or store
                        your full card details. Prices are in USD and exclude any applicable taxes/VAT, which are added
                        at checkout where required.
                    </p>
                </Section>

                <Section title="Cancellation & refunds">
                    <p>
                        You can cancel anytime from your account (or Stripe's billing portal). On cancellation your plan
                        stays active until the end of the current paid period; we do not charge you again after that.
                    </p>
                    <p>
                        Except where required by law, payments are <strong className="text-white">non-refundable</strong>{' '}
                        and we do not prorate partial periods. To avoid being charged for a period you don't want, cancel
                        before it renews (and, for the trial, before it ends).
                    </p>
                    <p>
                        <strong className="text-white">EU/EEA consumers:</strong> the Service is digital content/services
                        supplied immediately. By starting to use it (including during the trial) you request immediate
                        performance and acknowledge that you lose the 14-day right of withdrawal once performance has
                        begun, to the extent permitted by law.
                    </p>
                </Section>

                <Section title="You are responsible for what you upload">
                    <p>
                        Before processing a video you must confirm — via the checkbox in the upload interface — that you
                        own the content or have the rights to process it. By doing so you represent and warrant that:
                    </p>
                    <ul className="list-disc pl-6 space-y-1">
                        <li>You own all rights to the content, or have a valid license or permission to process it;</li>
                        <li>The content does not infringe any third-party copyright, trademark, privacy, or other right;</li>
                        <li>The content is not unlawful, defamatory, or otherwise prohibited.</li>
                    </ul>
                    <p>
                        If you submit content you do not have rights to, that is your responsibility. You agree to
                        indemnify OpenShorts and its contributors against any third-party claim arising from content you
                        submitted. We may suspend or terminate accounts that abuse the Service or infringe others' rights.
                    </p>
                </Section>

                <Section title="What we store, and for how long">
                    <ul className="list-disc pl-6 space-y-1">
                        <li>
                            <strong className="text-white">Account data:</strong> your email address and your subscription
                            status and usage (minutes used). Kept while your account exists.
                        </li>
                        <li>
                            <strong className="text-white">Generated videos:</strong> the clips you create are stored in
                            encrypted cloud storage and available in your library <strong className="text-white">while your
                            subscription is active, plus 7 days after it ends</strong>, then permanently deleted.
                        </li>
                        <li>
                            <strong className="text-white">Uploaded/source files &amp; working data:</strong> deleted from
                            our processing servers shortly after the job finishes (typically within 1 hour).
                        </li>
                        <li>
                            <strong className="text-white">Billing data:</strong> handled by Stripe; we keep a reference
                            to your Stripe customer/subscription, not your card.
                        </li>
                        <li>
                            <strong className="text-white">Optional add-on keys (ElevenLabs, fal.ai):</strong> for BYOK
                            features, stored encrypted in your browser and sent as request headers only when needed —
                            never written to our database.
                        </li>
                        <li>
                            <strong className="text-white">Server access logs:</strong> retained up to 30 days for
                            debugging and abuse prevention.
                        </li>
                    </ul>
                    <p>We do not sell, rent, or share your data with third parties for advertising or any unrelated purpose.</p>
                </Section>

                <Section title="Subprocessors">
                    <p>To provide the hosted Service we share the minimum necessary data with a small number of
                        trusted service providers, each acting on our behalf:</p>
                    <ul className="list-disc pl-6 space-y-1">
                        <li><strong className="text-white">A payments provider</strong> — payments &amp; subscriptions.</li>
                        <li><strong className="text-white">A cloud infrastructure &amp; storage provider</strong> — hosting and storing your generated videos.</li>
                        <li><strong className="text-white">An AI provider</strong> — video analysis, titles and thumbnails.</li>
                        <li><strong className="text-white">A social-publishing provider</strong> — posting to TikTok, Instagram &amp; YouTube (only when you connect them).</li>
                        <li><strong className="text-white">An email provider</strong> — transactional email (sign-in links, notices).</li>
                    </ul>
                    <p>Each is bound by its own terms and privacy policy. We can identify the specific providers to
                        you on request where required by law.</p>
                </Section>

                <Section title="Service is provided as-is">
                    <p>
                        The Service is provided on a best-effort basis with no warranties of any kind and no guarantee of
                        uptime, accuracy, or fitness for a particular purpose. To the maximum extent permitted by law, our
                        aggregate liability is limited to the amount you paid us in the 3 months before the claim, and we
                        are not liable for indirect or consequential damages. Availability of specific features may vary
                        over time and is not guaranteed.
                    </p>
                </Section>

                <Section title="Your rights (EU / EEA / UK)">
                    <p>
                        Under the GDPR / UK GDPR you may access, rectify, erase, restrict, object to, or port your
                        personal data. You can delete your account and its data — including your video library — by
                        emailing{' '}
                        <a className="text-primary underline" href={`mailto:${SUPPORT_EMAIL}`}>{SUPPORT_EMAIL}</a>. You may
                        also lodge a complaint with your local supervisory authority (in Spain: AEPD,{' '}
                        <a className="text-primary underline" href="https://www.aepd.es" target="_blank" rel="noopener noreferrer">
                            aepd.es
                        </a>
                        ).
                    </p>
                </Section>

                <Section title="Copyright takedowns">
                    <p>
                        If you believe content processed through the Service infringes your copyright, email{' '}
                        <a className="text-primary underline" href={`mailto:${SUPPORT_EMAIL}`}>{SUPPORT_EMAIL}</a> with:
                        identification of the work, identification of the allegedly infringing material (enough detail to
                        locate it), your contact information, and a statement that you are authorized to act for the
                        rights holder.
                    </p>
                </Section>

                <Section title="Self-hosted instances">
                    <p>
                        OpenShorts is open source and may be self-hosted. This notice applies to the hosted version we
                        operate at openshorts.app. Self-hosted instances are run by their administrators, whose data
                        handling and policies are their own responsibility, not ours.
                    </p>
                </Section>

                <Section title="Changes & contact">
                    <p>
                        We may update this notice; the "Last updated" date reflects the latest revision. For material
                        changes affecting paid subscribers we'll give reasonable notice. Continued use after a change
                        constitutes acceptance. Questions:{' '}
                        <a className="text-primary underline" href={`mailto:${SUPPORT_EMAIL}`}>{SUPPORT_EMAIL}</a> or{' '}
                        <a className="text-primary underline" href={ISSUES_URL} target="_blank" rel="noopener noreferrer">
                            GitHub Issues
                        </a>
                        .
                    </p>
                    <p>These terms are governed by the laws of Spain.</p>
                </Section>
            </main>
        </div>
    );
}
