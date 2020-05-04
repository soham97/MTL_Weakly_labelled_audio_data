labels = ['Acoustic_guitar', 'Applause', 'Bark', 'Bass_drum', 
          'Burping_or_eructation', 'Bus', 'Cello', 'Chime', 'Clarinet', 
          'Computer_keyboard', 'Cough', 'Cowbell', 'Double_bass', 
          'Drawer_open_or_close', 'Electric_piano', 'Fart', 'Finger_snapping', 
          'Fireworks', 'Flute', 'Glockenspiel', 'Gong', 'Gunshot_or_gunfire', 
          'Harmonica', 'Hi-hat', 'Keys_jangling', 'Knock', 'Laughter', 'Meow', 
          'Microwave_oven', 'Oboe', 'Saxophone', 'Scissors', 'Shatter', 
          'Snare_drum', 'Squeak', 'Tambourine', 'Tearing', 'Telephone', 
          'Trumpet', 'Violin_or_fiddle', 'Writing']

lb_to_ix = {lb: i for i, lb in enumerate(labels)}
ix_to_lb = {i: lb for i, lb in enumerate(labels)}

seq_len = 311
mel_bins = 64
classes_num = 41
