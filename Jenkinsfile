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
                sh 'docker build -t placement-app -f docker/Dockerfile .'
            }
        }

        stage('Run Docker Container') {
            steps {
                sh 'docker stop placement-container || true'
                sh 'docker rm placement-container || true'
                sh 'docker run -d -p 5001:5000 --name placement-container placement-app'
            }
        }

    }
}