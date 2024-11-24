from ilqr_solver import LogBarrieriLQR, NNiLQR, ModelBasediLQR
from scenario.car_parking import CarParking
from scenario.quadcopter import QuadCopter
from scenario.robotic_arm_tracking import RoboticArmTracking
from utils.Logger import logger
import matplotlib.pyplot as plt

def train_and_save(scenario, folder_name, noise_std=0.0, use_noise=False, bias=0.0):
    logger.set_folder_name(folder_name, remove_existing_folder=True)
    logger.set_is_use_logger(True)
    logger.set_is_save_json(True)
    nn_ilqr = NNiLQR(gaussian_noise_sigma=[[0.1], [0.1], [0.1], [0.1]], iLQR_max_iter=100)
    nn_ilqr.init(scenario, use_noise=use_noise, noise_std=noise_std).solve()
    scenario.play(folder_name)

if __name__ == "__main__":
    # for i in range(10):
    #     try:
    #         logger.set_folder_name("QuadCopter_" + str(i), remove_existing_folder=True) #여기 False로 바꿔야
    #         logger.set_is_use_logger(True)
    #         logger.set_is_save_json(True)
    #         scenario = QuadCopter() 
    #         NNiLQR(gaussian_noise_sigma=[[0.1], [0.1], [0.1], [0.1]], iLQR_max_iter=100).init(scenario).solve() 
    #     except Exception as e:
    #         logger.error(f"An error occurred during processing: {str(e)}")
    #     continue
    # scenario = QuadCopter() 
    # scenario.play("QuadCopter_9")

    # logger.set_folder_name("QuadCopter_Log", remove_existing_folder=True).set_is_use_logger(True).set_is_save_json(True)
    # scenario = QuadCopter() 
    # LogBarrieriLQR().init(scenario).solve() 
    # scenario.play("QuadCopter_Log")

    # #1개만 추출해서하기
    # logger.set_folder_name("QuadCopter_" + str(0), remove_existing_folder=True)
    # logger.set_is_use_logger(True)
    # logger.set_is_save_json(True)
    # scenario = QuadCopter()
    # NNiLQR(gaussian_noise_sigma=[[0.1], [0.1], [0.1], [0.1]], iLQR_max_iter=100).init(scenario, use_noise = False, noise_std = 0.0).solve()
    # folder_name = "QuadCopter_" + str(0)
    # saved_data = logger.read_from_json(folder_name, no_iter=-1)  # 마지막 iteration의 trajectory 읽기
    # trajectory = saved_data["trajectory"]
    # print(f"Trajectory from {folder_name}:")
    # print(trajectory)
    # scenario.play("QuadCopter_0")

    # Quadcopter_1 파일은 noisy training한 거 확인하기
    # logger.set_folder_name("QuadCopter_" + str(1), remove_existing_folder=True)
    # logger.set_is_use_logger(True)
    # logger.set_is_save_json(True)
    # scenario = QuadCopter()
    # NNiLQR(gaussian_noise_sigma=[[0.1], [0.1], [0.1], [0.1]], iLQR_max_iter=100).init(scenario, use_noise = True, noise_std = 0.001).solve()
    # scenario.play("QuadCopter_1")

    scenario = QuadCopter()

    #Noise 없는 경우
    # train_and_save(scenario, "QuadCopter_0", noise_std=0.0, use_noise=False)

    # #Noise 있는 경우
    # train_and_save(scenario, "QuadCopter_1", noise_std=0.001, use_noise=True)
    # train_and_save(scenario, "QuadCopter_2", noise_std=0.005, use_noise=True)
    train_and_save(scenario, "QuadCopter_3", noise_std=0.002, use_noise=True)
    # train_and_save(scenario, "QuadCopter_9", noise_std=0.003, use_noise=True)
    # train_and_save(scenario, "QuadCopter_10", noise_std=0.004, use_noise=True)
    # train_and_save(scenario, "QuadCopter_11", noise_std=0.0005, use_noise=True)
    # train_and_save(scenario, "QuadCopter_14", noise_std=0.005, use_noise=True)

    # #Bias 있는 경우
    # train_and_save(scenario, "QuadCopter_4", noise_std=0.0, use_noise=True, bias=0.0005)
    # train_and_save(scenario, "QuadCopter_5", noise_std=0.001, use_noise=True, bias=0.0005)
    # train_and_save(scenario, "QuadCopter_6", noise_std=0.001, use_noise=True, bias=0.001)
    # train_and_save(scenario, "QuadCopter_7", noise_std=0.001, use_noise=True, bias=0.002)
    # train_and_save(scenario, "QuadCopter_8", noise_std = 0.001, use_noise = True, bias = 0.005)
    # train_and_save(scenario, "QuadCopter_12", noise_std = 0.001, use_noise = True, bias = 0.007)
    # train_and_save(scenario, "QuadCopter_13", noise_std = 0.001, use_noise = True, bias = 0.01)

    # #Plot results
    # scenario.plot_noisy_train(["QuadCopter_0", "QuadCopter_1", "QuadCopter_2", "QuadCopter_3"])
    # scenario.plot_noisy_train(["QuadCopter_0", "QuadCopter_4", "QuadCopter_5", "QuadCopter_6"])

    # model-based
    # logger.set_folder_name("QuadCopter_model_based", remove_existing_folder=True)  # 추가
    # logger.set_is_use_logger(True)
    # logger.set_is_save_json(True)
    # model_based_ilqr = ModelBasediLQR(iLQR_max_iter=100)
    # model_based_ilqr.init(scenario).solve()


# if __name__ == "__main__":
#     for i in range(10):
#         try:
#             logger.set_folder_name("CarParking_" + str(i), remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#             scenario = CarParking() 
#             NNiLQR(gaussian_noise_sigma=[[0.01], [0.1]], iLQR_max_iter=100).init(scenario).solve() 
#         except Exception as e:
#             pass
#         continue
#     scenario = CarParking() 
#     scenario.play("CarParking_9")

#     logger.set_folder_name("CarParking_Log", remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#     scenario = CarParking() 
#     LogBarrieriLQR().init(scenario).solve() 
#     scenario.play("CarParking_Log")

# if __name__ == "__main__":
#     for i in range(10):
#         try:
#             logger.set_folder_name("RoboticArmTracking_" + str(i), remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#             scenario = RoboticArmTracking() 
#             NNiLQR(gaussian_noise_sigma=[[0.1], [0.1]], iLQR_max_iter=100, training_stopping_criterion=0.01, decay_rate_max_iters=200).init(scenario).solve() 
#         except Exception as e:
#             pass
#         continue
#     scenario = RoboticArmTracking() 
#     scenario.play("RoboticArmTracking_9")

#     logger.set_folder_name("RoboticArmTracking_log", remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#     scenario = RoboticArmTracking() 
#     LogBarrieriLQR().init(scenario).solve() 
#     scenario.play("RoboticArmTracking_log")
#     plt.show()