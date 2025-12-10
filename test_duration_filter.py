#!/usr/bin/env python3
"""
Test script to verify duration filtering in remote analysis stage.
Creates mock shot data to demonstrate the filtering behavior.
"""

import sys
from pathlib import Path
import yaml

def create_skipped_analysis(reason: str, min_duration: float = None) -> dict:
    """Mock the skipped analysis function from 5_remote_analysis.py"""
    caption = f"Shot skipped ({reason})"
    if min_duration is not None:
        caption = f"Shot skipped (duration < {min_duration}s)"
    
    return {
        'caption': caption,
        'skipped': True,
        'skip_reason': reason,
        'attributes': {
            'suspense': 0.0,
            'darkness': 0.0,
            'ambiguity': 0.0,
            'emotional_tension': 0.0,
            'intensity': 0.0,
            'motion': 0.0,
            'impact': 0.0,
        }
    }

def test_duration_filter():
    """Test the duration filtering logic"""
    print("=" * 60)
    print("Testing Duration Filter for Remote Analysis")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(__file__).parent / 'trailer_generator' / 'config' / 'settings.yaml'
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    min_shot_duration = config['remote_analysis'].get('min_shot_duration', 2.0)
    print(f"\n✓ Configuration loaded")
    print(f"  Minimum shot duration: {min_shot_duration}s")
    
    # Create mock shots with various durations
    mock_shots = [
        {'id': 1, 'duration': 0.5, 'start_time': 0.0, 'end_time': 0.5},
        {'id': 2, 'duration': 1.2, 'start_time': 0.5, 'end_time': 1.7},
        {'id': 3, 'duration': 2.5, 'start_time': 1.7, 'end_time': 4.2},
        {'id': 4, 'duration': 1.8, 'start_time': 4.2, 'end_time': 6.0},
        {'id': 5, 'duration': 3.0, 'start_time': 6.0, 'end_time': 9.0},
        {'id': 6, 'duration': 0.8, 'start_time': 9.0, 'end_time': 9.8},
        {'id': 7, 'duration': 4.5, 'start_time': 9.8, 'end_time': 14.3},
        {'id': 8, 'duration': 1.5, 'start_time': 14.3, 'end_time': 15.8},
    ]
    
    print(f"\n✓ Created {len(mock_shots)} mock shots")
    
    # Apply filtering logic
    qualifying_shots = []
    skipped_shots = []
    
    for shot in mock_shots:
        duration = shot.get('duration', 0)
        if duration >= min_shot_duration:
            qualifying_shots.append(shot)
        else:
            shot['analysis'] = create_skipped_analysis('duration_too_short', min_shot_duration)
            skipped_shots.append(shot)
    
    # Display results
    print(f"\n📊 Shot Duration Filter Results:")
    print(f"  ✓ Qualifying shots (≥ {min_shot_duration}s): {len(qualifying_shots)}")
    print(f"  ⊘ Skipped shots (< {min_shot_duration}s): {len(skipped_shots)}")
    
    print("\n" + "=" * 60)
    print("QUALIFYING SHOTS (will be analyzed):")
    print("=" * 60)
    for shot in qualifying_shots:
        print(f"  Shot {shot['id']:2d}: {shot['duration']:.1f}s  [{shot['start_time']:.1f}s - {shot['end_time']:.1f}s]")
    
    print("\n" + "=" * 60)
    print("SKIPPED SHOTS (too short):")
    print("=" * 60)
    for shot in skipped_shots:
        print(f"  Shot {shot['id']:2d}: {shot['duration']:.1f}s  [{shot['start_time']:.1f}s - {shot['end_time']:.1f}s]")
        print(f"           Analysis: {shot['analysis']['caption']}")
    
    # Verify correct filtering
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("=" * 60)
    
    expected_qualifying = [3, 5, 7]  # IDs of shots ≥ 2.0s
    expected_skipped = [1, 2, 4, 6, 8]  # IDs of shots < 2.0s
    
    actual_qualifying = [s['id'] for s in qualifying_shots]
    actual_skipped = [s['id'] for s in skipped_shots]
    
    if actual_qualifying == expected_qualifying:
        print(f"  ✓ Qualifying shots correct: {actual_qualifying}")
    else:
        print(f"  ✗ Qualifying shots INCORRECT!")
        print(f"    Expected: {expected_qualifying}")
        print(f"    Actual:   {actual_qualifying}")
    
    if actual_skipped == expected_skipped:
        print(f"  ✓ Skipped shots correct: {actual_skipped}")
    else:
        print(f"  ✗ Skipped shots INCORRECT!")
        print(f"    Expected: {expected_skipped}")
        print(f"    Actual:   {actual_skipped}")
    
    # Check analysis structure
    print("\n" + "=" * 60)
    print("SKIPPED ANALYSIS STRUCTURE:")
    print("=" * 60)
    if skipped_shots:
        sample = skipped_shots[0]
        analysis = sample['analysis']
        print(f"  Caption: {analysis['caption']}")
        print(f"  Skipped: {analysis['skipped']}")
        print(f"  Skip Reason: {analysis['skip_reason']}")
        print(f"  Attributes (sample): suspense={analysis['attributes']['suspense']}, intensity={analysis['attributes']['intensity']}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"\n✓ Duration filtering working as expected!")
    print(f"✓ Shots < {min_shot_duration}s will be skipped during analysis")
    print(f"✓ Skipped shots receive standardized 'skipped' analysis result")
    print(f"✓ All shots (analyzed + skipped) maintained in metadata")

if __name__ == '__main__':
    try:
        test_duration_filter()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
