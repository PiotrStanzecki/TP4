#include <SDL.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>
#include "simulate.h"
#include "matplot/matplot.h"

using namespace matplot;

// Function to initialize SDL
int init(std::shared_ptr<SDL_Window>& gWindow, std::shared_ptr<SDL_Renderer>& gRenderer, const int SCREEN_WIDTH, const int SCREEN_HEIGHT)
{
    if (SDL_Init(SDL_INIT_VIDEO) >= 0)
    {
        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
        gWindow = std::shared_ptr<SDL_Window>(SDL_CreateWindow("Planar Quadrotor", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN), SDL_DestroyWindow);
        gRenderer = std::shared_ptr<SDL_Renderer>(SDL_CreateRenderer(gWindow.get(), -1, SDL_RENDERER_ACCELERATED), SDL_DestroyRenderer);
        SDL_SetRenderDrawColor(gRenderer.get(), 0xFF, 0xFF, 0xFF, 0xFF);
    }
    else
    {
        std::cout << "SDL_ERROR: " << SDL_GetError() << std::endl;
        return -1;
    }
    return 0;
}

// Function to calculate LQR gain matrix
Eigen::MatrixXf LQR(PlanarQuadrotor &quadrotor, float dt) {
    Eigen::MatrixXf Eye = Eigen::MatrixXf::Identity(6, 6);
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(6, 6);
    Eigen::MatrixXf A_discrete = Eigen::MatrixXf::Zero(6, 6);
    Eigen::MatrixXf B(6, 2);
    Eigen::MatrixXf B_discrete(6, 2);
    Eigen::MatrixXf Q = Eigen::MatrixXf::Identity(6, 6);
    Eigen::MatrixXf R = Eigen::MatrixXf::Identity(2, 2);
    Eigen::MatrixXf K = Eigen::MatrixXf::Zero(6, 6);
    Eigen::Vector2f input = quadrotor.GravityCompInput();

    Q.diagonal() <<  0.004, 0.004, 400, 0.005, 0.045, 2 / 2 / M_PI;
    R.row(0) << 30, 7;
    R.row(1) << 7, 30;

    std::tie(A, B) = quadrotor.Linearize();
    A_discrete = Eye + dt * A;
    B_discrete = dt * B;
    
    return LQR(A_discrete, B_discrete, Q, R);
}

// Function to control the quadrotor
void control(PlanarQuadrotor &quadrotor, const Eigen::MatrixXf &K) {
    Eigen::Vector2f input = quadrotor.GravityCompInput();
    quadrotor.SetInput(input - K * quadrotor.GetControlState());
}

int main(int argc, char* args[])
{
    std::shared_ptr<SDL_Window> gWindow = nullptr;
    std::shared_ptr<SDL_Renderer> gRenderer = nullptr;
    const int SCREEN_WIDTH = 1280;
    const int SCREEN_HEIGHT = 720;

    Eigen::VectorXf initial_state = Eigen::VectorXf::Zero(6);
    PlanarQuadrotor quadrotor(initial_state);
    PlanarQuadrotorVisualizer quadrotor_visualizer(&quadrotor);

    Eigen::VectorXf goal_state = Eigen::VectorXf::Zero(6);
    goal_state << -1, 7, 0, 0, 0, 0;
    quadrotor.SetGoal(goal_state);

    const float dt = 0.01;
    Eigen::MatrixXf K = LQR(quadrotor, dt);
    Eigen::Vector2f input = Eigen::Vector2f::Zero(2);

    std::vector<float> x_history;
    std::vector<float> y_history;
    std::vector<float> theta_history;

    auto f = figure(true);
    f->size(800, 600);
    auto ax1 = subplot(3, 1, 1);
    auto l1 = plot(ax1, x_history);
    title(ax1, "x trajectory");

    auto ax2 = subplot(3, 1, 2);
    auto l2 = plot(ax2, y_history);
    title(ax2, "y trajectory");

    auto ax3 = subplot(3, 1, 3);
    auto l3 = plot(ax3, theta_history);
    title(ax3, "theta trajectory");

    if (init(gWindow, gRenderer, SCREEN_WIDTH, SCREEN_HEIGHT) >= 0)
    {
        SDL_Event e;
        bool quit = false;
        float delay;
        int x, y;
        Eigen::VectorXf state = Eigen::VectorXf::Zero(6);
        int plot_counter = 0;

        while (!quit)
        {
            // Event handling
            while (SDL_PollEvent(&e) != 0)
            {
                if (e.type == SDL_QUIT)
                {
                    quit = true;
                }
                else if (e.type == SDL_MOUSEBUTTONDOWN)
                {
                    SDL_GetMouseState(&x, &y);
                    goal_state << x, y, 0, 0, 0, 0;
                    quadrotor.SetGoal(goal_state);
                    std::cout << "Mouse position: (" << x << ", " << y << ")" << std::endl;
                }
                else if (e.type == SDL_KEYDOWN)
                {
                    if (e.key.keysym.sym == SDLK_p)
                    {
                        // Plot trajectory
                        ax1->clear();
                        ax1->hold(true);
                        plot(ax1, x_history);
                        title(ax1, "x trajectory");

                        ax2->clear();
                        ax2->hold(true);
                        plot(ax2, y_history);
                        title(ax2, "y trajectory");

                        ax3->clear();
                        ax3->hold(true);
                        plot(ax3, theta_history);
                        title(ax3, "theta trajectory");

                        f->draw();
                    }
                }
            }

            SDL_Delay((int)(dt * 1000));

            SDL_SetRenderDrawColor(gRenderer.get(), 0xFF, 0xFF, 0xFF, 0xFF);
            SDL_RenderClear(gRenderer.get());

            /* Simulate quadrotor forward in time */
            control(quadrotor, K);
            quadrotor.Update(dt);

            /* Store the current state for plotting */
            state = quadrotor.GetState();
            x_history.push_back(state(0));
            y_history.push_back(state(1));
            theta_history.push_back(state(2));

            // Limit the size of history vectors to 1000
            if (x_history.size() > 1000) {
                x_history.erase(x_history.begin());
            }
            if (y_history.size() > 1000) {
                y_history.erase(y_history.begin());
            }
            if (theta_history.size() > 1000) {
                theta_history.erase(theta_history.begin());
            }

            // Quadrotor rendering step
            quadrotor_visualizer.render(gRenderer);

            // Get the current state
            state = quadrotor.GetState();
            float cx = state(0); // Center x
            float cy = state(1); // Center y
            float theta = state(2); // Angle theta

            // Define the length of the line (adjust as needed)
            float line_length = 100.0f;

            // Calculate the endpoints of the line
            float x1 = cx - 0.5f * line_length * cos(theta);
            float y1 = cy - 0.5f * line_length * sin(theta);
            float x2 = cx + 0.5f * line_length * cos(theta);
            float y2 = cy + 0.5f * line_length * sin(theta);

            

            // Draw the line
            SDL_SetRenderDrawColor(gRenderer.get(), 0x00, 0x00, 0x00, 0xFF); // Black color
            SDL_RenderDrawLine(gRenderer.get(), x1, y1, x2, y2);

            SDL_RenderPresent(gRenderer.get());
        }
    }
    SDL_Quit();
    return 0;
}
