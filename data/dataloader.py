import numpy as np

class DataLoader_UCIHAR:
    def __init__(self, data_path):
        self.data_path = data_path
        self.activity_labels = {
            0: 'WALKING',
            1: 'WALKING_UPSTAIRS',
            2: 'WALKING_DOWNSTAIRS',
            3: 'SITTING',
            4: 'STANDING',
            5: 'LAYING'
        }

    def load_signals(self, subset):
        signals_path = f'{self.data_path}/{subset}/Inertial Signals/'
        body_acc_x = np.loadtxt(f'{signals_path}body_acc_x_{subset}.txt')
        body_acc_y = np.loadtxt(f'{signals_path}body_acc_y_{subset}.txt')
        body_acc_z = np.loadtxt(f'{signals_path}body_acc_z_{subset}.txt')
        body_gyro_x = np.loadtxt(f'{signals_path}body_gyro_x_{subset}.txt')
        body_gyro_y = np.loadtxt(f'{signals_path}body_gyro_y_{subset}.txt')
        body_gyro_z = np.loadtxt(f'{signals_path}body_gyro_z_{subset}.txt')
        X = np.stack([
            body_acc_x, body_acc_y, body_acc_z,
            body_gyro_x, body_gyro_y, body_gyro_z
        ], axis=2)
        return X

    def load_labels(self, subset):
        y = np.loadtxt(f'{self.data_path}/{subset}/y_{subset}.txt')
        y = y - 1
        return y.astype(int)

    def load_all_data(self):
        print("Loading training data...")
        X_train = self.load_signals('train')
        y_train = self.load_labels('train')
        print("Loading test data...")
        X_test = self.load_signals('test')
        y_test = self.load_labels('test')
        print(f"Train: X={X_train.shape}, y={y_train.shape}")
        print(f"Test: X={X_test.shape}, y={y_test.shape}")
        return X_train, y_train, X_test, y_test

    def get_data_statistics(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"\nTraining samples: {len(y_train)}")
        print(f"Test samples: {len(y_test)}")
        print(f"Time steps per sample: {X_train.shape[1]}")
        print(f"Features per timestep: {X_train.shape[2]}")
        print("\nClass Distribution (Training):")
        for label, name in self.activity_labels.items():
            count = np.sum(y_train == label)
            percentage = (count / len(y_train)) * 100
            print(f"  {label} - {name:20s}: {count:4d} ({percentage:.1f}%)")
        print("\nClass Distribution (Test):")
        for label, name in self.activity_labels.items():
            count = np.sum(y_test == label)
            percentage = (count / len(y_test)) * 100
            print(f"  {label} - {name:20s}: {count:4d} ({percentage:.1f}%)")
