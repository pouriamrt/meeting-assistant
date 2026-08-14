# Meeting assistant

Hears both sides of a call, transcribes it on your own machine, and writes cue
cards into a window that screen capture cannot see.

## Before you use it

**This records both sides of a conversation.** Whether that is legal without
telling the other person depends on where you and they are. Some places allow
it if one participant consents, which is you; others require everyone to agree.
Check before you point this at a real call.

**The overlay is invisible to screen sharing.** That is the entire point of the
feature, and it is also why using it in an interview or a proctored exam is
normally a breach of that organisation's rules, whatever the law says. Your
call, but make it deliberately.

It was written for personal use. Nobody is monitoring how you use it, and
nobody is standing behind it either.

## Setup

1. Unzip anywhere. Run `meeting-assistant.exe`.
2. Windows will warn that it is unsigned, because it is. "More info" then "Run
   anyway", or don't, if you would rather not.
3. It asks once for an Anthropic API key. Get one at
   <https://console.anthropic.com>. Answers are billed to that key, roughly
   $0.03 a press.
4. First launch downloads a speech model, about 141 MB, once. It needs internet
   for that and says so while it happens.
5. **Use headphones.** On speakers the microphone hears the other person coming
   out of them and every question gets transcribed twice, which breaks the main
   hotkey.

## Keys

| Key | Does |
|---|---|
| `Ctrl+Space` | Answer the last question they asked |
| `Ctrl+Shift+Space` | Answer a question you type |
| `Ctrl+Alt+Space` | Read the screen and answer what's on it, in code |
| `Ctrl+Alt+H` | Hide or show the overlay |
| `Ctrl+Alt+Q` | Quit |

Drag the bar along the top to move it, the corner square to resize.
`Ctrl`+scroll changes text size, `Shift`+scroll makes it see-through. Size and
position are remembered.

Lost it behind another window? Press `Ctrl+Alt+H` twice.

## Requirements

Windows 10 build 19041 or newer. No GPU needed. Transcription runs on your CPU
and the audio never leaves your machine; only the resulting text goes to
Anthropic, plus a screenshot when you press `Ctrl+Alt+Space`.

## What it does not hide

Screen capture and screen sharing, yes. A capture card, a phone camera pointed
at the monitor, or somebody sitting next to you, no.

## When something breaks

`%LOCALAPPDATA%\meeting-assistant\assistant.log` has the details. Your key is
in the same folder, encrypted against your Windows account. Delete
`ui.json` there to reset the window, `key.bin` to be asked for a key again.
