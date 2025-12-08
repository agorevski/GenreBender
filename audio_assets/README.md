# Audio Assets Directory

This directory contains audio resources for trailer generation.

## Structure

```
audio_assets/
├── music/          # Background music tracks
│   └── (place music files here)
└── effects/        # Sound effects (future use)
    └── (place sound effect files here)
```

## Music Library

### Supported Formats
- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- AAC (.aac)
- OGG (.ogg)

### Naming Convention

For automatic genre matching, include genre keywords in filenames:

**Thriller:**
- `thriller_suspense_01.mp3`
- `dark_tension_music.wav`
- `suspenseful_track.mp3`

**Action:**
- `action_epic_01.mp3`
- `intense_battle_music.wav`
- `adrenaline_rush.mp3`

**Horror:**
- `horror_atmospheric_01.mp3`
- `creepy_ambient.wav`
- `scary_tension.mp3`

**Drama:**
- `drama_emotional_01.mp3`
- `touching_piano.wav`
- `heartfelt_strings.mp3`

**Sci-Fi:**
- `scifi_futuristic_01.mp3`
- `electronic_space.wav`
- `tech_ambient.mp3`

**Comedy:**
- `comedy_upbeat_01.mp3`
- `quirky_fun.wav`
- `cheerful_music.mp3`

**Romance:**
- `romance_sweet_01.mp3`
- `romantic_strings.wav`
- `love_theme.mp3`

### AI Music Selection

When AI music selection is enabled (`ai_music_selection: true` in settings.yaml):
- The system will analyze your timeline and recommend tracks
- It considers pacing, mood, and narrative structure
- Falls back to tag-based selection if AI fails

### Manual Music Selection

You can specify a music file directly when running stage 9:

```bash
python 9_audio_mixing.py --input movie.mp4 --genre thriller --music-file /path/to/your/music.mp3
```

## Getting Started

1. Add music files to the `music/` directory
2. Name files with genre keywords for better matching
3. Run stage 9 to generate final trailer with audio

## Free Music Resources

Consider these royalty-free music sources:
- YouTube Audio Library
- Free Music Archive
- Incompetech
- Bensound
- Purple Planet

**Note:** Always verify licensing rights before using music in your projects.
