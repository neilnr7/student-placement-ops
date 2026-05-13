pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                echo 'Cloning repository...'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t placement-app -f docker/Dockerfile .'
            }
        }

        stage('Run Docker Container') {
            steps {
                bat 'docker stop placement-container || exit 0'
                bat 'docker rm placement-container || exit 0'
                bat 'docker run -d -p 5001:5000 --name placement-container placement-app'
            }
        }

    }
}