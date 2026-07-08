# Whisper kept thanking me for messages I never sent

I'm building [Handsfree](https://handsfree.contraststudio.app), an iOS app that lets you talk to Claude Code by voice — your phone connects to a small bridge on your own machine, you speak, the agent codes, and it reads the replies back. A few weeks ago I started adding a better transcription path, and Whisper spent the next several days politely thanking me for things I never said.

This is the story of three hallucination bugs, in increasing order of weirdness, and what actually fixed them.

## The setup: hybrid transcription

Apple's on-device speech recognition is fast and free, but the transcripts are raw: no punctuation, filler words everywhere, and mixed-language speech comes out mangled. Whisper produces beautiful transcripts — punctuated, multilingual — but it's a batch model, not a streaming one.

So I did both:

- **On-device** (`SFSpeechRecognizer`) provides the live transcript while you talk, plus endpoint detection — it decides when you've stopped speaking.
- On endpoint, the phone ships the **raw utterance audio** (16-bit PCM over WebSocket) to the bridge, where **whisper.cpp** (large-v3-turbo, Metal, via [smart-whisper](https://github.com/JacobLinCool/smart-whisper)) transcribes the whole thing with full context.
- The polished Whisper transcript is what actually gets sent to the agent. If the bridge is slow or down, the on-device text is the fallback.

First tests were great. Punctuation, capitalization, `settings` and `fix` preserved as English inside Chinese sentences. Then I stopped talking for a moment.

## Bug 1: the phantom "Thank you."

During testing, I dismissed a half-spoken message and paused. The app sent **"Thank you."** to my coding agent. I didn't say anything. I tried to cancel another message — it sent *another* "Thank you."

The bridge log made the pattern obvious:

```
18:29:31 [stt.transcript] chars=10 ms=1203 text=Thank you.
18:29:48 [stt.transcript] chars=10 ms=1184 text=Thank you.
```

Both "utterances" were near-silent audio buffers — leftover capture from after a dismissed message. Whisper looked at silence and confidently produced gratitude.

This turns out to be one of Whisper's most famous failure modes. It was trained on a vast amount of YouTube-style audio with subtitles, and the tail end of that training data is saturated with outro lines. Feed it silence or near-silence and the decoder's most probable continuation is subtitle boilerplate: *"Thank you."*, *"Thanks for watching!"*, *"Please subscribe."* There are long GitHub threads about it ([openai/whisper #679](https://github.com/openai/whisper/discussions/679), [whisper.cpp #1724](https://github.com/ggml-org/whisper.cpp/issues/1724)) and even [a paper investigating hallucinations induced by non-speech audio](https://arxiv.org/abs/2501.11378).

**Fix 1: don't ask Whisper about silence.** I already had a second opinion available — the on-device recognizer. If `SFSpeechRecognizer` heard zero words in the buffer, it's silence; discard it and never call Whisper at all. The on-device transcript became a speech-presence gate:

```swift
// The on-device transcript is our speech-presence gate. If SFSpeech heard
// nothing, it's silence — do NOT hand the buffer to Whisper, which
// HALLUCINATES ("Thank you", "Thanks for watching", …) on silence.
guard !onDeviceText.isEmpty else {
    discardCapturedAudio()
    return
}
```

Phantom messages: gone. For about an hour.

## Bug 2: "Thank you. Thank you." — now at the *start*

The next hallucination arrived at the *beginning* of a real message:

> **Thank you. Thank you.** I think this is working. Next, I'm going to test some other languages…

I definitely didn't open with double gratitude. The log had the tell:

```
19:06:43 [stt.trim] fromMs=123500 toMs=107010
```

The audio buffer was **123 seconds long**. Why? Barge-in. The app lets you interrupt the agent mid-reply, and to catch your first word, it starts recording while the reply is still playing. That reply played for two minutes before I interrupted — and the whole time, the buffer was recording the *echo-cancelled residue* of the app's own text-to-speech.

Hardware echo cancellation is good, but its output isn't digital zero — it's a faint smear of almost-silence. So Whisper received: **~100 seconds of ghostly near-silence, then me talking.** It transcribed the ghost first. The ghost says thank you.

**Fix 2 came in two parts:**

1. **Client:** at the moment of barge-in, throw away everything recorded during the reply except a 2-second pre-roll (enough to keep the user's first words, which is what the recording was for).
2. **Bridge:** trim leading/trailing near-silence before every transcription — RMS energy over 30 ms windows, drop everything below a threshold, keep ~120 ms of margin at the front so word onsets don't get clipped and ~60 ms at the tail:

```ts
const win = 480;        // 30 ms @ 16 kHz
const thresh = 0.012;   // RMS gate: speech sits well above, room noise below
// … find first/last voiced window, keep small pads, subarray the rest away
```

Now Whisper only ever saw actual speech with tight margins. Surely that's the end of it.

## Bug 3: the Chinese subtitle volunteer

I switched testing to Mandarin. At the end of an ordinary message, Whisper appended:

> 中文字幕志愿者 杨栋梁

That translates to roughly *"Chinese subtitle volunteer: Yang Dongliang."* Whisper didn't just hallucinate a phrase — it **credited a fictional member of the fansub team** that transcribed my nonexistent video.

It's the same YouTube-training-data failure mode wearing a different hat: in Chinese content, the boilerplate at the end of subtitled videos isn't "thanks for watching," it's the subtitle group's credit line. Different language, same statistical ghost.

But wait — I was trimming silence now. The log agreed:

```
19:57:28 [stt.trim] fromMs=23500 toMs=19680   ← trimmed 3.8s
19:57:31 [stt.transcript] …都给删了吧。中文字幕志愿者 杨栋梁
```

The trim ran, and the hallucination survived. The problem: **an energy gate is not a voice activity detector.** The tail of that recording wasn't digital silence — it was breath and faint mouth noise, *just* above the RMS threshold. Loud enough to survive the trim; speech-free enough to trigger the hallucination. You can't fix that by tuning the threshold, because the next recording's breath will sit somewhere else.

**Fix 3: make the model police itself.** whisper.cpp exposes decoder-level knobs for exactly this, and they're the standard anti-hallucination stack:

```ts
const opts = {
  language,
  suppress_non_speech_tokens: true,  // ban non-speech tokens outright
  no_speech_thold: 0.6,              // trust the model's own "no speech here" signal
  entropy_thold: 2.4,                // reject high-entropy (rambling) segments
  logprob_thold: -1.0,               // reject low-confidence segments
};
```

Whisper computes a `no_speech` probability and per-token confidences internally — by default it just doesn't act on them very aggressively. These thresholds tell it: if you suspect a segment isn't speech, or you're not confident in what you decoded, drop it instead of improvising.

## Bonus round: the punctuation primer

One more Whisper quirk from the same week, too good not to include. English transcripts came back beautifully punctuated. Chinese transcripts came back as one endless run-on sentence — unless I paused dramatically between phrases like a news anchor.

The fix is delightfully dumb: **give Whisper a punctuated example.** The `initial_prompt` parameter primes the decoder's context, and the decoder imitates the style of what it sees:

```ts
function punctuationPrimer(language: string): string {
  switch (language) {
    case "zh": return "以下是普通话的句子，包含逗号、句号和问号等标点符号。";
    case "ja": return "以下は、句読点を含む日本語の文章です。";
    default:   return ""; // English punctuates fine on its own
  }
}
```

One punctuated Mandarin sentence as a primer, and every Chinese transcript came back with proper 逗号 and 句号. (It also nudges the output toward Simplified characters, which I wanted anyway.)

## What I'd tell you if you're shipping Whisper in a product

1. **Never feed Whisper silence.** It doesn't return empty — it returns YouTube outros. Gate with real speech detection *before* the model. If you have any secondary signal that speech occurred (another recognizer, a VAD), use it as a hard gate.
2. **Energy trimming is necessary but not sufficient.** It removes the obvious padding, but breath and room tone sail right through an RMS gate, and they hallucinate just as well as silence does.
3. **Turn on the decoder's own defenses.** `suppress_non_speech_tokens` and the `no_speech`/`entropy`/`logprob` thresholds exist because the model already *knows* when it's improvising. Make it act on that.
4. **Prime for punctuation in CJK languages.** A one-sentence punctuated `initial_prompt` fixes run-on Chinese transcripts almost completely.
5. **Log everything about the audio, not just the text.** Every one of these bugs was solved by one log line — buffer duration, trim amounts, transcription time. `fromMs=123500` told the whole barge-in story instantly.

The hallucinations were never random. Each one was Whisper doing exactly what it was trained to do — completing the most statistically likely subtitle for the audio it was given. The bug was always in what I was giving it.

---

*Handsfree is a free iOS app for talking to Claude Code and Codex hands-free — the agent runs on your own machine, and your phone talks to it over your own network. It's [on the App Store](https://apps.apple.com/app/apple-store/id6781863812?pt=128939791&ct=blog&mt=8), and the bridge is `npx handsfree-bridge` ([MIT, open source](https://github.com/devtheicetea/handsfree)).*
